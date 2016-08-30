import numba
import numpy as np


class ePModel:
    '''
    Simple leak IF model with additional function to compute the filter 
    and equivalent input current of the eP model used to compute the sensitivity to the field.
    '''
    def __init__(self):
        # Set the parameters of the initial BS model.
        self.rhom_dend  = 28.0e-1   # Dendritic tree membrane resistivity (Ohm m2)
        self.rha_dend   = 1.5       # Dendritic tree membrane internal resistivity (Ohm m)
        self.d_dend     = 1.2e-6    # Dendritic tree diameter (m)
        self.L          = 700.0e-6  # Dendritic tree length (m)

        # Soma (here we take the same values for the soma and dendritic tree as was done in Rattay 1999)
        self.C_soma     = 1.0e-2   # Soma membrane capacitance (F / m2)
        self.rhom_soma  = 28.0e-1  # Soma membrane resistivity (Ohm m2)
        self.d_soma     = 10.e-6    # Soma diameter ( m )

        self.updateAuxParams()

        # Firing parameters 
        self.V_cut      = 10e-3    # Voltage cut for the firing (V)
        self.T_ref      = 1.5e-3    # Refractory time (s)
        self.V_reset    = 5.e-3    # Reset membrane potential (V)
        
        self.EL     = 0.    # V
        
    def updateAuxParams(self):
        '''
        This function should be called everytime some parameters of the model are changed.
        It is used to update the auxiliary parameters, e.g. parameters derived directly from the default moel parameters. 
        '''
        # Cable resistance per unit length
        self.r_m    = self.rhom_dend / ( np.pi * self.d_dend )        # Tangential resistance
        self.r_i    = self.rha_dend / ( np.pi * (self.d_dend/2) ** 2) # Axial resistance

        # Cable specific length
        self.R_s    = self.rhom_soma / ( np.pi * self.d_soma ** 2 )
        self.C_s    = self.C_soma * np.pi * self.d_soma ** 2
        self.lambd  = np.sqrt( self.r_m / self.r_i ) 
        self.gamma  = self.R_s / ( self.r_i * self.lambd )
        self.tau_m  = self.R_s * self.C_s
        
        self.gL     = 1 / self.R_s
        
    def BSComplexPolarization(self,freqs):
        '''
        Compute the BS somatic membrane polarization due to an electrical field of 1V/m
        
        Parameters:
            :param freqs:   Numpy array. Field frequency (Hz) for which we want to compute the polarization
            
        :returns:
            Numpy array of complex float corresponding to the BS somatic membrane polarization
        '''
        w = 2 * np.pi * freqs

        Rs = self.rhom_soma / ( np.pi * self.d_soma ** 2 )
        Cs = self.C_soma * np.pi * self.d_soma ** 2 
        Gs = 1 / Rs
        gm = 1 / self.r_m
        cm = self.C_soma * (self.d_dend * np.pi)
        gi = 1 / self.r_i

        alpha   = np.sqrt( ( gm     + np.sqrt( gm**2 + w**2 * cm**2 ) ) / ( 2 * gi ) )
        beta    = np.sqrt( ( -gm    + np.sqrt( gm**2 + w**2 * cm**2 ) ) / ( 2 * gi ) )
        z       = alpha + 1.j*beta  # z2 = -z

        dummy   = 1 + np.exp(-2 * z * self.L)
        denom   = dummy * ( Cs * w * 1.j + Gs) + z * gi * ( 2 - dummy )
        #V1_BS_over_I1(iw) = dummy / denom
        return gi * ( 2 * np.exp( -z * self.L ) - dummy ) / denom

    def fieldEqInputCurrent(self,freqs):
        '''
        Return the input current equivalent to the extracellular electrical field for given field frequencies
        The equivalent input current is complex, its imaginary part should be taken in case of a sinusoidal field
        '''
        # We just need to compute the factor which is due to the time integration of the current
        return  self.BSComplexPolarization(freqs) / self.computePointNeuronImpedance(freqs)

    def runSim(self,duration=500e-3,E_onset=100e-3,E_amp=1,E_freq=10,E_n=1000,V0=-65e-3,
               dt=0.1e-3,I_ext=None,t_spikes=None): 
        '''
        Run a time integration of the IF neuron model to which the input current equivalent to the 
        field is added.
        
        Parameters:
        :param duration:    Simulation duration (s)
        :param offset:      Time at which the field start (s)
        :param E_amp:       Field amplitude (V/m)
        :param E_freq:      Field frequency (Hz)
        :param V0:          Base Membrane potential (resting value when no field is applied) (V)
        :param I_ext:       External input received by the neuron at the soma  (A)
                                Numpy.array of length (duration/dt)
        :param t_spikes:    (Output) List in which the spike time are stored (s)

        :returns:   (t,V,E): 3 numpy.array of length duration/dt corresponding to:
                            - the time values (s)
                            - the membrane voltage ( V )
                            - the field intensity ( V/m )
        '''        
        # Time sample array
        t = np.arange(0,duration,dt)
        nT = len(t)
        
        # Precompute the input current equivalent to the field
        E_comp = np.zeros((nT,),dtype=np.complex128)
        E_offset = E_onset + 1.0 * E_n / E_freq
        E_comp[(t > E_onset) & (t < E_offset)] = E_amp * np.exp(2 * np.pi * E_freq 
                                                                * ( t[(t>E_onset) & (t < E_offset)]
                                                                   - t[t>E_onset][0])
                                                                * 1j 
                                                               )
        
        IE = np.imag(E_comp * self.fieldEqInputCurrent(E_freq))
        E = np.imag(E_comp)
        
        V = np.zeros(E.shape)
        V[0] = V0


        T_refIndex = int(np.ceil(self.T_ref / dt))
       
        spike_mask = np.zeros_like(V)
        ePModel._integrationIF(V,IE,nT, \
                                            self.C_s,self.gL,self.EL,\
                                            dt,I_ext,self.V_reset,self.V_cut,T_refIndex,spike_mask)
       
        if not t_spikes is None:
            t_spikes_arr = t[np.nonzero(spike_mask)]
            t_spikes.extend(t_spikes_arr.tolist())


        return t,V,E

    @numba.njit
    def _integrationIF(V,IE,nT,C_s,gL,EL,dt,I_ext,V_reset,V_cut,T_refIndex,spike_mask):
        '''
        Internal time integration for the model using Numba for faster computation
        '''
        lastFiringIndex = -T_refIndex
        
        for i in xrange(nT-1):
            
            # Are we in the refractory time?
            if i - lastFiringIndex < T_refIndex:
                V[i+1] = V_reset
                continue
                       
            V[i+1] = V[i] + dt / C_s * (-gL * ( V[i]-EL ) + IE[i] + I_ext[i])
            
            if V[i+1] > V_cut:
                spike_mask[i+1] = 1
                lastFiringIndex = i+1
                V[i+1] = V_reset

    def computeEquivalentInputCurrentFromBSInput(self,t_BS,I_BS,V0=0):
        '''
        Compute the equivalent input current to apply on the point neuron model to reproduce the oscillations
        observed in the B&S model for an input current at the soma

        WARNING:
            this is an analytical solutions mvalid oly for the passive case:
                - linearization of the exp mech.
                - Slow adaptation limit
            

        Parameters:
            :param t_BS:    Time stamp corresponding to I_BS (s)
            :param I_BS:    Input current applied at the B&S soma (A)
            :param V0:      Base membrane potential (V)

        :returns:   I_Point: numpy.array containing the time serie of the equivalent input current to apply
                                on the point neuron (A)

        '''
        # Decompose the input current on the BS model into its fourier coefs
        fcoefs_I_BS = np.fft.fft(I_BS)
        freqs = np.fft.fftfreq(len(I_BS),np.diff(t_BS)[0])
            
        # Compute the membrane potential oscillations due to the input currents in the fourier space
        factors_BS = self.computeBSImpedance_soma(freqs)
        fcoefs_V = fcoefs_I_BS * factors_BS

        # Compute the equivalent input current for Point neuron model in the foureier space
        factors_PointNeuron = 1/self.computePointNeuronImpedance(freqs)
    
        fcoefs_I_Point = fcoefs_V * factors_PointNeuron

        # Reconstitute the input current from its foureier coefs
        I_Point = np.real( np.fft.ifft(fcoefs_I_Point) )

        return I_Point

    def computePointNeuronImpedance(self,freqs):
        '''
        Compute the point neuron impedance.

        Parameters:
            :param freqs:   Frequencies for which the impedance should be computed (Hz)lt

        :returns:   Complex impedances
        '''
        return 1/(self.gL +  2 * 1.j * np.pi * freqs * self.C_s)


    def computeBSImpedance_soma(self, freqs):
            '''
            Compute the BS complex impedance at the soma, i.e. the somatic membrane voltage
            change due to a somatic input.
            
            Parameters:
                :param freqs:  Input frequency for which the impedance is evaluated

            :returns:
                Impedance in V/A
            '''
            w = 2 * np.pi * freqs
            
            Gs = 1 / self.R_s
            gm = 1 / self.r_m
            cm = self.C_soma * (self.d_dend * np.pi)
            gi = 1 / self.r_i

            alpha   = np.sqrt( ( gm     + np.sqrt( gm**2 + w**2 * cm**2 ) ) / ( 2 * gi ) )
            beta    = np.sqrt( ( -gm    + np.sqrt( gm**2 + w**2 * cm**2 ) ) / ( 2 * gi ) )
            z       = alpha + 1.j*beta  # z2 = -z
            z[w<0]  = np.conj(z[w<0])

            dummy   = 1 + np.exp(-2 * z * self.L)
            denom   = dummy * ( self.C_s * w * 1.j + Gs) + z * gi * ( 2 - dummy )
            return dummy / denom

    def computeEquivalentInputCurrentFromBSInput_dend(self,t_BS,I_BS,V0=0):
        '''
        Compute the equivalent input current to apply on the point neuron model to reproduce the oscillations
        observed in the B&S model for an input current at the dendritic end 

        WARNING:
            this is an analytical solutions mvalid oly for the passive case:
                - linearization of the exp mech.
                - Slow adaptation limit
            

        Parameters:
            :param t_BS:    Time stamp corresponding to I_BS (s)
            :param I_BS:    Input current applied at the B&S soma (A)
            :param V0:      Base membrane potential (V)

        :returns:   I_Point: numpy.array containing the time serie of the equivalent input current to apply
                                on the point neuron (A)

        '''
        # Decompose the input current on the BS model into its fourier coefs
        fcoefs_I_BS = np.fft.fft(I_BS)
        freqs = np.fft.fftfreq(len(I_BS),np.diff(t_BS)[0])
            
        # Compute the membrane potential oscillations due to the input currents in the fourier space
        factors_BS = self.computeBSImpedance_dend(freqs)
        fcoefs_V = fcoefs_I_BS * factors_BS

        # Compute the equivalent input current for Point neuron model in the foureier space
        factors_PointNeuron = 1/self.computePointNeuronImpedance(freqs)
    
        fcoefs_I_Point = fcoefs_V * factors_PointNeuron

        # Reconstitute the input current from its foureier coefs
        I_Point = np.real( np.fft.ifft(fcoefs_I_Point) )

        return I_Point

    def computeBSImpedance_dend(self, freqs):
            '''
            Compute the BS complex impedance at the dendritic end, i.e. the somatic membrane voltage
            change due to a somatic input.
            
            Parameters:
                :param freqs:  Input frequency for which the impedance is evaluated

            :returns:
                Impedance in V/A
            '''
            w = 2 * np.pi * freqs
            
            Gs = 1 / self.R_s
            gm = 1 / self.r_m
            cm = self.C_soma * (self.d_dend * np.pi)
            gi = 1 / self.r_i

            alpha   = np.sqrt( ( gm     + np.sqrt( gm**2 + w**2 * cm**2 ) ) / ( 2 * gi ) )
            beta    = np.sqrt( ( -gm    + np.sqrt( gm**2 + w**2 * cm**2 ) ) / ( 2 * gi ) )
            z       = alpha + 1.j*beta  # z2 = -z
            z[w<0]  = np.conj(z[w<0])

            num     = 1 
            denom   =  ( ( Gs + 1.j * w * self.C_s ) * np.cosh(z * self.L) + gi * z * np.sinh(z * self.L) )

            return num / denom