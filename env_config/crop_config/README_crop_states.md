# README_crop_params.md
# An overview of all the available crop 
# State and Rate values for output to the simulation model

**############################################################################**
# WOFOST States and Rates
**############################################################################**
**State variables:** (For output to observation space):
============  ================================================= ==== ===============
 Name          Description                                      Pbl      Unit
============  ================================================= ==== ===============
TAGP          Total above-ground Production                      N    |kg ha-1|
GASST         Total gross assimilation                           N    |kg CH2O ha-1|
MREST         Total gross maintenance respiration                N    |kg CH2O ha-1|
CTRAT         Total crop transpiration accumulated over the
                crop cycle                                       N    cm
CEVST         Total soil evaporation accumulated over the
                crop cycle                                       N    cm
HI            Harvest Index (only calculated during              N    -
                finalize())
DOF           Date representing the day of finish of the crop    N    -
                simulation.
FINISH_TYPE   String representing the reason for finishing the   N    -
                simulation: maturity, harvest, leave death, etc.
============  ================================================= ==== ===============
**Rate variables:** (For output to observation space):
=======  ================================================ ==== =============
 Name     Description                                      Pbl      Unit
=======  ================================================ ==== =============
GASS     Assimilation rate corrected for water stress       N  |kg CH2O ha-1 d-1|
PGASS    Potential assimilation rate                        N  |kg CH2O ha-1 d-1|
MRES     Actual maintenance respiration rate, taking into
            account that MRES <= GASS.                      N  |kg CH2O ha-1 d-1|
PMRES    Potential maintenance respiration rate             N  |kg CH2O ha-1 d-1|
ASRC     Net available assimilates (GASS - MRES)            N  |kg CH2O ha-1 d-1|
DMI      Total dry matter increase, calculated as ASRC
            times a weighted conversion efficieny.          Y  |kg ha-1 d-1|
ADMI     Aboveground dry matter increase                    Y  |kg ha-1 d-1|
=======  ================================================ ==== =============

**############################################################################**
# Assimilation States and Rates
**############################################################################**

**############################################################################**
# Evapotranspiration States and Rates
**############################################################################**
**State variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
IDWST     Nr of days with water stress.                     N    -
IDOST     Nr of days with oxygen stress.                    N    -
=======  ================================================= ==== ============
**Rate variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
EVWMX    Maximum evaporation rate from an open water        Y    |cm day-1|
            surface.
EVSMX    Maximum evaporation rate from a wet soil surface.  Y    |cm day-1|
TRAMX    Maximum transpiration rate from the plant canopy   Y    |cm day-1|
TRA      Actual transpiration rate from the plant canopy    Y    |cm day-1|
IDOS     Indicates water stress on this day (True|False)    N    -
IDWS     Indicates oxygen stress on this day (True|False)   N    -
RFWS     Reducation factor for water stress                 Y     -
RFOS     Reducation factor for oxygen stress                Y     -
RFTRA    Reduction factor for transpiration (wat & ox)      Y     -
=======  ================================================= ==== ============

**############################################################################**
# Leaf Dynamics States and Rates
**############################################################################**
**State variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
LV       Leaf biomass per leaf class                        N    |kg ha-1|
SLA      Specific leaf area per leaf class                  N    |ha kg-1|
LVAGE    Leaf age per leaf class                            N    |d|
LVSUM    Sum of LV                                          N    |kg ha-1|
LAIEM    LAI at emergence                                   N    -
LASUM    Total leaf area as sum of LV*SLA,                  N    -
            not including stem and pod area                 
LAIEXP   LAI value under theoretical exponential growth     N    -
LAIMAX   Maximum LAI reached during growth cycle            N    -
LAI      Leaf area index, including stem and pod area       Y    -
WLV      Dry weight of living leaves                        Y    |kg ha-1|
DWLV     Dry weight of dead leaves                          N    |kg ha-1|
TWLV     Dry weight of total leaves (living + dead)         Y    |kg ha-1|
=======  ================================================= ==== ============
**Rate variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
GRLV     Growth rate leaves                                 N   |kg ha-1 d-1|
DSLV1    Death rate leaves due to water stress              N   |kg ha-1 d-1|
DSLV2    Death rate leaves due to self-shading              N   |kg ha-1 d-1|
DSLV3    Death rate leaves due to frost kill                N   |kg ha-1 d-1|
DSLV4    Death rate leaves due to nutrient stress           N   |kg ha-1 d-1|
DSLV     Maximum of DLSV1, DSLV2, DSLV3                     N   |kg ha-1 d-1|
DALV     Death rate leaves due to aging.                    N   |kg ha-1 d-1|
DRLV     Death rate leaves as a combination of DSLV and     N   |kg ha-1 d-1|
            DALV
SLAT     Specific leaf area for current time step,          N   |ha kg-1|
            adjusted for source/sink limited leaf expansion
            rate.
FYSAGE   Increase in physiological leaf age                 N   -
GLAIEX   Sink-limited leaf expansion rate (exponential      N   |ha ha-1 d-1|
            curve)
GLASOL   Source-limited leaf expansion rate (biomass        N   |ha ha-1 d-1|
            increase)
=======  ================================================= ==== ============

**############################################################################**
# NPK Dynamics States and Rates
**############################################################################**
**State variables** (For output to observation space):
==========  ================================================== ============
 Name        Description                                          Unit
==========  ================================================== ============
NamountLV     Actual N amount in living leaves                  |kg N ha-1|
PamountLV     Actual P amount in living leaves                  |kg P ha-1|
KamountLV     Actual K amount in living leaves                  |kg K ha-1|
    
NamountST     Actual N amount in living stems                   |kg N ha-1|
PamountST     Actual P amount in living stems                   |kg P ha-1|
KamountST     Actual K amount in living stems                   |kg K ha-1|

NamountSO     Actual N amount in living storage organs          |kg N ha-1|
PamountSO     Actual P amount in living storage organs          |kg P ha-1|
KamountSO     Actual K amount in living storage organs          |kg K ha-1|

NamountRT     Actual N amount in living roots                   |kg N ha-1|
PamountRT     Actual P amount in living roots                   |kg P ha-1|
KamountRT     Actual K amount in living roots                   |kg K ha-1|

Nuptake_T    total absorbed N amount                            |kg N ha-1|
Puptake_T    total absorbed P amount                            |kg P ha-1|
Kuptake_T    total absorbed K amount                            |kg K ha-1|
Nfix_T       total biological fixated N amount                  |kg N ha-1|
==========  ================================================== ============
**Rate variables** (For output to observation space):
===========  =================================================  ================
 Name         Description                                           Unit
===========  =================================================  ================
RNamountLV     Weight increase (N) in leaves                    |kg N ha-1 d-1|
RPamountLV     Weight increase (P) in leaves                    |kg P ha-1 d-1|
RKamountLV     Weight increase (K) in leaves                    |kg K ha-1 d-1|

RNamountST     Weight increase (N) in stems                     |kg N ha-1 d-1|
RPamountST     Weight increase (P) in stems                     |kg P ha-1 d-1|
RKamountST     Weight increase (K) in stems                     |kg K ha-1 d-1|
    
RNamountRT     Weight increase (N) in roots                     |kg N ha-1 d-1|
RPamountRT     Weight increase (P) in roots                     |kg P ha-1 d-1|
RKamountRT     Weight increase (K) in roots                     |kg K ha-1 d-1|

RNamountSO     Weight increase (N) in storage organs            |kg N ha-1 d-1|
RPamountSO     Weight increase (P) in storage organs            |kg P ha-1 d-1|
RKamountSO     Weight increase (K) in storage organs            |kg K ha-1 d-1|

RNdeathLV      Rate of N loss in leaves                         |kg N ha-1 d-1|
RPdeathLV      Rate of P loss in leaves                         |kg N ha-1 d-1|
RKdeathLV      Rate of K loss in leaves                         |kg N ha-1 d-1|

RNdeathST      Rate of N loss in roots                          |kg N ha-1 d-1|
RPdeathST      Rate of P loss in roots                          |kg N ha-1 d-1|
RKdeathST      Rate of K loss in roots                          |kg N ha-1 d-1|

RNdeathRT      Rate of N loss in stems                          |kg N ha-1 d-1|
RPdeathRT      Rate of P loss in stems                          |kg N ha-1 d-1|
RKdeathRT      Rate of K loss in stems                          |kg N ha-1 d-1|

RNloss         N loss due to senescence                         |kg N ha-1 d-1|
RPloss         P loss due to senescence                         |kg P ha-1 d-1|
RKloss         K loss due to senescence                         |kg K ha-1 d-1|
===========  =================================================  ================

**############################################################################**
# Partitioning States and Rates
**############################################################################**
**State variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
FR        Fraction partitioned to roots.                     Y    -
FS        Fraction partitioned to stems.                     Y    -
FL        Fraction partitioned to leaves.                    Y    -
FO        Fraction partitioned to storage orgains            Y    -
=======  ================================================= ==== ============

**############################################################################**
# Vernalization States and Rates
**############################################################################**
**State variables** (For output to observation space):
============ ================================================= ==== ========
 Name        Description                                       Pbl   Unit
============ ================================================= ==== ========
VERN         Vernalisation state                                N    days
DOV          Day when vernalisation requirements are            N    -
                fulfilled.
ISVERNALISED Flag indicated that vernalisation                  Y    -
                requirement has been reached
============ ================================================= ==== ========
**Rate variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
VERNR    Rate of vernalisation                              N     -
VERNFAC  Reduction factor on development rate due to        Y     -
            vernalisation effect.
=======  ================================================= ==== ============

**############################################################################**
# Phenology States and Rates
**############################################################################**
**State variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
DVS      Development stage                                  Y    - 
TSUM     Temperature sum                                    N    |C| day
TSUME    Temperature sum for emergence                      N    |C| day
DOS      Day of sowing                                      N    - 
DOE      Day of emergence                                   N    - 
DOA      Day of Anthesis                                    N    - 
DOM      Day of maturity                                    N    - 
DOH      Day of harvest                                     N    -
STAGE    Current phenological stage, can take the           N    -
            folowing values:
            emerging|vegetative|reproductive|mature
DSNG     Days since no crop growth (perennial only)         Y    day
DSD      Days since dormancy started (perennial only)       Y    day
AGE      Age of the crop in years (perennial only)          Y    year
DOP      Day of Planting                                    Y    -
DATBE    Number of consecutive days above temperature sum   Y    day
          for emergence
=======  ================================================= ==== ============
**Rate variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
DTSUME   Increase in temperature sum for emergence          N    |C|
DTSUM    Increase in temperature sum for anthesis or        N    |C|
            maturity
DVR      Development rate                                   Y    |day-1|
RDEM     Day increase when day temp is above TSUMEM         Y    day
=======  ================================================= ==== ============

**############################################################################**
# Respiration Dynamics States and Rates
**############################################################################**
**Rate variables:** (For output to observation space):
=======  ================================================ ==== =============
 Name     Description                                      Pbl      Unit
=======  ================================================ ==== =============
PMRES    Potential maintenance respiration rate             N  |kg CH2O ha-1 d-1|
=======  ================================================ ==== =============

**############################################################################**
# Root Dynamics States and Rates
**############################################################################**
**State variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
RD       Current rooting depth                              Y     cm
RDM      Maximum attainable rooting depth at the minimum    N     cm
            of the soil and crop maximum rooting depth
WRT      Weight of living roots                             Y     |kg ha-1|
DWRT     Weight of dead roots                               N     |kg ha-1|
TWRT     Total weight of roots                              Y     |kg ha-1|
=======  ================================================= ==== ============
**Rate variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
RR       Growth rate root depth                             N    cm
GRRT     Growth rate root biomass                           N   |kg ha-1 d-1|
DRRT1    Death rate of roots due to aging                   N   |kg ha-1 d-1|
DRRT2    Death rate of roots due to excess water            N   |kg ha-1 d-1|
DRRT3    Death rate of roots due to excess NPK              N   |kg ha-1 d-1|
DRRT     Death rate root biomass                            N   |kg ha-1 d-1|
GWRT     Net change in root biomass                         N   |kg ha-1 d-1|
=======  ================================================= ==== ============

**############################################################################**
# Stem Dynamics States and Rates
**############################################################################**
**State variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
SAI      Stem Area Index                                    Y     -
WST      Weight of living stems                             Y     |kg ha-1|
DWST     Weight of dead stems                               N     |kg ha-1|
TWST     Total weight of stems                              Y     |kg ha-1|
=======  ================================================= ==== ============
**Rate Variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
GRST     Growth rate stem biomass                           N   |kg ha-1 d-1|
DRST     Death rate stem biomass                            N   |kg ha-1 d-1|
GWST     Net change in stem biomass                         N   |kg ha-1 d-1|
=======  ================================================= ==== ============

**############################################################################**
# Storage Organs Dynamics States and Rates
**############################################################################**
**State variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
PAI      Pod Area Index                                     Y     -
WSO      Weight of living storage organs                    Y     |kg ha-1|
DWSO     Weight of dead storage organs                      N     |kg ha-1|
TWSO     Total weight of storage organs                     Y     |kg ha-1|
HWSO     Harvestable weight of storage organs               Y     |kg ha-1|
LHW      Last harvest weight of storage organs              Y     |kg ha-1|
=======  ================================================= ==== ============
**Rate variables** (For output to observation space):
=======  ================================================= ==== ============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ============
GRSO     Growth rate storage organs                         N   |kg ha-1 d-1|
DRSO     Death rate storage organs                          N   |kg ha-1 d-1|
DHSO     Death rate of harvestable storage organs           N   |kg ha-1 d-1|
GWSO     Net change in storage organ biomass                N   |kg ha-1 d-1|
=======  ================================================= ==== ============

**############################################################################**
# NPK Demand Uptake States and Rates
**############################################################################**
**State variables** (For output to observation space):
=============  ================================================= ==== ============
 Name           Description                                      Pbl      Unit
=============  ================================================= ==== ============
NuptakeTotal     Total N uptake by the crop                        N   |kg N ha-1|
PuptakeTotal     Total P uptake by the crop                        N   |kg N ha-1|
KuptakeTotal     Total K uptake by the crop                        N   |kg N ha-1|
NfixTotal        Total N fixated by the crop                       N   |kg N ha-1|

NdemandST        N Demand in living stems                          N   |kg N ha-1|
NdemandRT        N Demand in living roots                          N   |kg N ha-1|
NdemandSO        N Demand in storage organs                        N   |kg N ha-1|

PdemandLV        P Demand in living leaves                         N   |kg P ha-1|
PdemandST        P Demand in living stems                          N   |kg P ha-1|
PdemandRT        P Demand in living roots                          N   |kg P ha-1|
PdemandSO        P Demand in storage organs                        N   |kg P ha-1|

KdemandLV        K Demand in living leaves                         N   |kg K ha-1|
KdemandST        K Demand in living stems                          N   |kg K ha-1|
KdemandRT        K Demand in living roots                          N   |kg K ha-1|
KdemandSO        K Demand in storage organs                        N   |kg K ha-1|
==========  ================================================= ==== ============
**Rate variables** (For output to observation space):
===========  ================================================= ==== ================
 Name         Description                                      Pbl      Unit
===========  ================================================= ==== ================
RNuptakeLV     Rate of N uptake in leaves                        Y   |kg N ha-1 d-1|
RNuptakeST     Rate of N uptake in stems                         Y   |kg N ha-1 d-1|
RNuptakeRT     Rate of N uptake in roots                         Y   |kg N ha-1 d-1|
RNuptakeSO     Rate of N uptake in storage organs                Y   |kg N ha-1 d-1|

RPuptakeLV     Rate of P uptake in leaves                        Y   |kg P ha-1 d-1|
RPuptakeST     Rate of P uptake in stems                         Y   |kg P ha-1 d-1|
RPuptakeRT     Rate of P uptake in roots                         Y   |kg P ha-1 d-1|
RPuptakeSO     Rate of P uptake in storage organs                Y   |kg P ha-1 d-1|

RKuptakeLV     Rate of K uptake in leaves                        Y   |kg K ha-1 d-1|
RKuptakeST     Rate of K uptake in stems                         Y   |kg K ha-1 d-1|
RKuptakeRT     Rate of K uptake in roots                         Y   |kg K ha-1 d-1|
RKuptakeSO     Rate of K uptake in storage organs                Y   |kg K ha-1 d-1|

RNuptake       Total rate of N uptake                            Y   |kg N ha-1 d-1|
RPuptake       Total rate of P uptake                            Y   |kg P ha-1 d-1|
RKuptake       Total rate of K uptake                            Y   |kg K ha-1 d-1|
RNfixation     Rate of N fixation                                Y   |kg N ha-1 d-1|

NdemandLV      N Demand in living leaves                         N   |kg N ha-1|
NdemandST      N Demand in living stems                          N   |kg N ha-1|
NdemandRT      N Demand in living roots                          N   |kg N ha-1|
NdemandSO      N Demand in storage organs                        N   |kg N ha-1|

PdemandLV      P Demand in living leaves                         N   |kg P ha-1|
PdemandST      P Demand in living stems                          N   |kg P ha-1|
PdemandRT      P Demand in living roots                          N   |kg P ha-1|
PdemandSO      P Demand in storage organs                        N   |kg P ha-1|

KdemandLV      K Demand in living leaves                         N   |kg K ha-1|
KdemandST      K Demand in living stems                          N   |kg K ha-1|
KdemandRT      K Demand in living roots                          N   |kg K ha-1|
KdemandSO      K Demand in storage organs                        N   |kg K ha-1|

Ndemand        Total crop N demand                               N   |kg N ha-1 d-1|
Pdemand        Total crop P demand                               N   |kg P ha-1 d-1|
Kdemand        Total crop K demand                               N   |kg K ha-1 d-1|
===========  ================================================= ==== ================
**############################################################################**
# NPK Stress States and Rates
**############################################################################**
**Rate variables** (For output to observation space):
=======  ================================================= ==== ==============
 Name     Description                                      Pbl      Unit
=======  ================================================= ==== ==============
NNI       Nitrogen nutrition index                          Y     -
PNI       Phosphorous nutrition index                       N     -
KNI       Potassium nutrition index                         N     -
NPKI      Minimum of NNI, PNI, KNI                          Y     -
RFNPK     Reduction factor for |CO2| assimlation            N     -
            based on NPKI and the parameter NLUE_NPK
=======  ================================================= ==== ==============

**############################################################################**
# NPK Dynamics States and Rates
**############################################################################**
 **State variables** (For output to observation space):
===================  ================================================= ===== ============
 Name                  Description                                      Pbl      Unit
===================  ================================================= ===== ============
NtranslocatableLV     Translocatable N amount in living leaves           N    |kg N ha-1|
PtranslocatableLV     Translocatable P amount in living leaves           N    |kg P ha-1|
KtranslocatableLV     Translocatable K amount in living leaves           N    |kg K ha-1|
NtranslocatableST     Translocatable N amount in living stems            N    |kg N ha-1|
PtranslocatableST     Translocatable P amount in living stems            N    |kg P ha-1|
KtranslocatableST     Translocatable K amount in living stems            N    |kg K ha-1|
NtranslocatableRT     Translocatable N amount in living roots            N    |kg N ha-1|
PtranslocatableRT     Translocatable P amount in living roots            N    |kg P ha-1|
KtranslocatableRT     Translocatable K amount in living roots            N    |kg K ha-1|
Ntranslocatable       Total N amount that can be translocated to the     Y    [kg N ha-1]
                        storage organs
Ptranslocatable       Total P amount that can be translocated to the     Y    [kg P ha-1]
                        storage organs
Ktranslocatable       Total K amount that can be translocated to the     Y    [kg K ha-1]
                        storage organs
===================  ================================================= ===== ============
**Rate variables** (For output to observation space):
===================  ================================================= ==== ==============
 Name                 Description                                      Pbl      Unit
===================  ================================================= ==== ==============
RNtranslocationLV     Weight increase (N) in leaves                     Y    |kg ha-1 d-1|
RPtranslocationLV     Weight increase (P) in leaves                     Y    |kg ha-1 d-1|
RKtranslocationLV     Weight increase (K) in leaves                     Y    |kg ha-1 d-1|
RNtranslocationST     Weight increase (N) in stems                      Y    |kg ha-1 d-1|
RPtranslocationST     Weight increase (P) in stems                      Y    |kg ha-1 d-1|
RKtranslocationST     Weight increase (K) in stems                      Y    |kg ha-1 d-1|
RNtranslocationRT     Weight increase (N) in roots                      Y    |kg ha-1 d-1|
RPtranslocationRT     Weight increase (P) in roots                      Y    |kg ha-1 d-1|
RKtranslocationRT     Weight increase (K) in roots                      Y    |kg ha-1 d-1|
===================  ================================================= ==== ==============
