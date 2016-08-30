COMMENT
fsin.mod

This file was shared by Ted Carnevale on the NEURON's forum:
http://www.neuron.yale.edu/phpbb/viewtopic.php?f=8&t=1814&start=15
http://www.neuron.yale.edu/ftp/ted/neuron/fsin.mod


A bogus point process that contains the variable x, 
which oscillates starting at t = del >= 0.
User specifies the frequency f of the oscillation 
and number of whole cycles n.

fsin uses the event delivery system to ensure compatibility with adaptive integration.

=================
NOTES AND CAVEATS
=================

1.  If x were a RANGE variable, an assignment statement would 
have to be inserted into proc advance() in order for the 
value of x to be used by other mechanisms--e.g.
proc advance() {
  is_xtra = Fsin[0].x
  fadvance()
}
However, that would be incompatible with adaptive integration.
To eliminate the need for such an assignment statement, x is a 
POINTER.  This preserves compatibility with adaptive integration.

2.  On every fadvance, the statements that evaluate Fsin's x 
should be executed before the statements in any client mechanism 
that relies on the value of Fsin's x.  To that end, the value of 
x is computed in a BEFORE BREAKPOINT block, which will take care
of any client mechanism that uses Fsin's x in a BREAKPOINT block.

However, some client mechanisms may have their own 
BEFORE BREAKPOINT blocks that need the value of Fsin's x.  
xtra is such a mechanism.  In this situation, care is required 
to ensure that the statements in Fsin's BEFORE BREAKPOINT block
are executed first.  This can be done by compiling the mod file 
that defines Fsin _before_ the client mechanism's mod file.

There are two ways to make this happen:
A.  Invoke nrnivmodl with a command line that presents the file 
names in the desired sequence.  UNIX/Linux users may be quite 
comfortable with this.
B.  Choose mod file names so that Fsin's mod file appears before 
the name of any client mod files in an alphabetical listing.
For the example of Fsin and xtra, the file names fsin.mod and 
xtra.mod would be quite suitable.  This is more convenient for 
users of all operating systems, but especially MSWin and OS X, 
whose users are accustomed to compiling all mod files in a 
directory with mknrndll or "drag and drop," respectively.

11/29/2009 NTC
ENDCOMMENT

NEURON {
  POINT_PROCESS Fsin
  RANGE del, f, amp, n, mean
  POINTER x
}

UNITS {
  PI = (pi) (1)
}

PARAMETER {
  del (ms)
  f (1/s)  : frequency is in Hz
  amp (1)
  n (1)
  mean (1): Mean input
}

ASSIGNED {
  x (1)
  on (1)
}

INITIAL {
  x = 0
  on = 0

  if (del<0) { del=0 }
  if (n<0) { n=0 }
  if (f<=0) { f=0 (1/s) }

  : do nothing if n or f == 0
  if ((n>0)&&(f>0)) {
    net_send(del, 1)  : to turn it on
    net_send(del+(n/f)*(1000), 1)  : to turn it off
  }
}

BEFORE BREAKPOINT {
  if (on==0) {
    x = 0
  } else {
    x = amp * sin(2*PI*(t-del)*f*(0.001)) + mean
  }
}

NET_RECEIVE (w) {
  : respond only to self-events with flag > 0
  if (flag == 1) {
    if (on==0) {
      on = 1  : turn it on
    } else {
      on = 0  : turn it off
    }
  }
}

COMMENT
for gcc 3.2.3
NET_RECEIVE (w) {
  : respond only to self-events with flag > 0
  if (flag != 0) {
VERBATIM
    on = (double)(on == 0.0);
ENDVERBATIM
  }
}
ENDCOMMENT
