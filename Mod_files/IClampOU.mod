COMMENT
-------------------------------------------------------------------------------
This code for the Ornstein-Uhlenbeck process is adapted from the code associated with Rudolph and Destexhe 2005:
https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=64259&file=\NCnote\Gfluct.mod

-------------------------------------------------------------------------------
ENDCOMMENT

NEURON {
    POINT_PROCESS IClampOU
    RANGE i, std, tau, mean
    ELECTRODE_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
}

PARAMETER {
    dt		(ms)

    mean    = 0     (nA)    : Mean value of the OU process
    std     = 1     (nA)    : Standard deviation of the OU process
    tau     = 5     (ms)    : Time constante of the OU process
}

ASSIGNED {
    i       (nA)
    mu
    D       (nA nA/ms)
    amp     (nA)
    ival    (nA)
}

INITIAL {
    i = mean
    ival = 0
    if (tau != 0) {
		D       = 2 * std * std / tau
		mu      = exp(-dt/tau)
		amp     = std * sqrt( (1-exp(-2*dt/tau)) )
	}
}


BEFORE BREAKPOINT {
    : The new noise value is computed before the breakpoint
    if( tau == 0 ) {
	   ival = std * normrand(0,1)
	} else {
    	ival = mu * ival + amp * normrand(0,1)
    }
}

BREAKPOINT {
    : The breakpoint is run twice during fadvance, therefore the new value of i should be first computed before the breakpoint
    : and then updated during the breakpoint
    : cf: http://www.neuron.yale.edu/phpBB/viewtopic.php?f=16&t=2055&p=7753&hilit=white+noise#p7753
    i = ival + mean
}

PROCEDURE seed(x) {
    set_seed(x)
}
