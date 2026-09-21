    UTCI, Version a 0.002, October 2009
    Copyright (C) 2009  Peter Broede

    Program for calculating UTCI Temperature (UTCI)
    released for public use after termination of COST Action 730

    replaces Version a 0.001, from September 2009

    Version History:

    pre-alpha version, October 2008:
    based on calculations as presented at the MC/WG meeting in Eilat, 2008-09-10

    Version a 0.001, September 2009:
    Changes compared to the pre-alpha version were based on decisions after the
    WG1 meeting in Stuttgart 2009-02-24:
     - Changed clothing model after WG1 meeting in Stuttgart 2009-02-24
     - Changed physiological model with respect to hidromeiosis
     - Equivalent temperature calculated by explicit comparisons
       to values of a 'Response Index' computed
       as first principal component from values after 30 and 120 min of 
       rectal temperature, mean skin temperature, face skin temperature,
       sweat rate, wetted skin area, shivering, skin blood flow.
     - as in the pre-alpha version, UTCI values are approximated by a 6th order polynomial regression function
     - The limits of mean radiant temperatures were extended to 30 degC below 
       and 70 degC above air temperature
     - Please note: The given polynomial approximation limits the application of
       this procedure to values of wind speed between 0.5 and 17 m/s!    

    Version a 0.002, October 2009
    Changed ReadMe text and program messages for public release

    Copyright (C) 2009  Peter Broede
    
    The programs are distributed in the hope that they will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  

The following files are distributed:

ReadMe_UTCI_a002.txt -- This file.
UTCI_a002.f90 -- FORTRAN source file for calculating UTCI from meteorological data entered via the keyboard.
UTCI_a002.exe -- WINDOWS executable generated from the FORTRAN source by the freely availble G95 compiler (http://g95.org).

METHOD:
A subroutine (UTCI_approx) was written that calculates UTCI values approximated by a 6th order polynomial from the input
 -- air temperature (-50 to +50 degC)
 -- mean radiant temperature (30 degC below to 70 degC above air temperature)
 -- wind speed in 10 m (0.5 to 17 m/s)
 -- water vapour presuure in hPa (below 50 hPa or 100% relative humidity)
Values in brackets indicate the range of validity.

USAGE:
This subroutine can be incorporated into your own applications, the programs provided here may serve as an instructive example.

DO NOT USE ANY PREVIOUS VERSION FOR PROFESSIONAL WORK ANY LONGER!!!

There is also a subroutine calculating saturation vapour pressure (es) used for converting relative humidity to vapour pressure and vice versa, that may be replaced by your own routines.

Installation:
The files are provided as a ZIP-archive. Store this archive into a temporary folder (e.g. C:\Temp). 
Unpacking the archive creates a subfolder (UTCI_a002) containing the files described above.
The programs have been compiled under MS WINDOWS 2000 (R), but should also run under newer versions.
The source code should also compile under Linux or Mac OS X (R) with the g95 compiler mentioned above, but this was not tested.


Disclaimer of Warranty.

THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR 
OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. 
SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

Limitation of Liability.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS THE 
PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR 
INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES 
OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH 
DAMAGES.

-----------------------------------------------
Peter Broede
Leibniz Research Centre for Working Environment and Human Factors (IfADo)
Leibniz-Institut für Arbeitsforschung an der TU Dortmund
Ardeystr. 67
D-44139 Dortmund
Fon: +49 +231 1084225
Fax: +49 +231 1084400
e-mail: broede@ifado.de
http://www.ifado.de
-----------------------------------------------
