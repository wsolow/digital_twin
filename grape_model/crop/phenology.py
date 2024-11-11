"""Implementation of  models for phenological development in WOFOST

Classes defined here:
- DVS_Phenology: Implements the algorithms for phenologic development
- Vernalisation: 

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""
import datetime

from ..utils.traitlets import Float, Instance, Enum, Bool, Int, Dict
from ..utils.decorators import prepare_rates, prepare_states

from ..util import limit, AfgenTrait, daylength
from ..base import ParamTemplate, StatesTemplate, RatesTemplate, \
     SimulationObject, VariableKiosk
from ..utils import signals
from ..utils import exceptions as exc
from ..nasapower import WeatherDataProvider

def daily_temp_units(drv: WeatherDataProvider, T0BC: float, TMBC: float):
    """
    Compute the daily temperature units using the BRIN model.
    Used for predicting budbreak in grapes.

    Slightly modified to not use the min temp at day n+1, but rather reuse the min
    temp at day n
    """
    A_c = 0
    for h in range(1, 25):
        # Perform linear interpolation between the hours 1 and 24
        if h <= 12:
            T_n = drv.TMIN + h * ((drv.TMAX - drv.TMIN) / 12)
        else:
            T_n = drv.TMAX - (h - 12) * ((drv.TMAX - drv.TMIN) / 12)

        # Limit the interpolation based on parameters
        T_n = limit(0, TMBC - T0BC, T_n - T0BC)
        A_c += T_n

    return A_c / 24          

class Grape_Phenology(SimulationObject):
    """Implements grape phenology based on many papers provided by Markus Keller
    at Washington State University
    
    Phenologic development in WOFOST is expresses using a unitless scale which
    takes the values 0 at emergence, 1 at Anthesis (flowering) and 2 at
    maturity. This type of phenological development is mainly representative
    for cereal crops. All other crops that are simulated with WOFOST are
    forced into this scheme as well, although this may not be appropriate for
    all crops. For example, for potatoes development stage 1 represents the
    start of tuber formation rather than flowering.
    
    Phenological development is mainly governed by temperature and can be
    modified by the effects of day length and vernalization 
    during the period before Anthesis. After Anthesis, only temperature
    influences the development rate.


    **Simulation parameters**
    
    =======  ============================================= =======  ============
     Name     Description                                   Type     Unit
    =======  ============================================= =======  ============
    TSUMEM   Temperature sum from sowing to emergence       SCr        |C| day
    TBASEM   Base temperature for emergence                 SCr        |C|
    TEFFMX   Maximum effective temperature for emergence    SCr        |C|
    TSUM1    Temperature sum from emergence to anthesis     SCr        |C| day
    TSUM2    Temperature sum from anthesis to maturity      SCr        |C| day
    TSUM3    Temperature sum form maturity to death         SCr        |C| day
    DVSI     Initial development stage at emergence.        SCr        -
             Usually this is zero, but it can be higher
             for crops that are transplanted (e.g. paddy
             rice)
    DVSEND   Final development stage                        SCr        -
    =======  ============================================= =======  ============

    **State variables**

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
             `emerging|vegetative|reproductive|mature`
    DATBE    Days above Temperature sum for emergence           N    days
    =======  ================================================= ==== ============

    **Rate variables**

    =======  ================================================= ==== ============
     Name     Description                                      Pbl      Unit
    =======  ================================================= ==== ============
    DTSUME   Increase in temperature sum for emergence          N    |C|
    DTSUM    Increase in temperature sum for anthesis or        N    |C|
             maturity
    DVR      Development rate                                   Y    |day-1|
    RDEM     Day counter for if day is suitable for germination Y    day
    =======  ================================================= ==== ============
    
    **External dependencies:**

    None    

    **Signals sent or handled**
    
    `DVS_Phenology` sends the `crop_finish` signal when maturity is
    reached and the `end_type` is 'maturity'.
    
    """

    _DAY_LENGTH = Float(12.0) # Helper variable for daylength
    _CU_FLAG    = Bool(False) # Helper flag for if chilling units should be accumulated
    _STAGE_VAL = Dict({"ecodorm":0, "budbreak":1, "flowering":2, "verasion":3, "ripe":4, "endodorm":5})

    class Parameters(ParamTemplate):
        CROP_START_TYPE = Enum(["predorm", "endodorm", "ecodorm"])
        CROP_END_TYPE = Enum(["endodorm", "ecodorm", "budbreak", "verasion", "ripe", "max_duration"])

        DVSI   = Float(-99.)  # Initial development stage
        DVSM   = Float(-99.)  # Mature development stage
        DVSEND = Float(-99.)  # Final development stage
        TBASEM = Float(-99.)  # Base temp. for bud break
        TEFFMX = Float(-99.)  # Max eff temperature for grow daily units
        TSUMEM = Float(-99.)  # Temp. sum for bud break

        TSUM1  = Float(-99.)  # Temperature sum budbreak to flowering
        TSUM2  = Float(-99.)  # Temperature sum flowering to verasion
        TSUM3  = Float(-99.)  # Temperature sum from verasion to ripe
        MLDORM = Float(-99.)  # Daylength at which a plant will go into dormancy
        Q10C   = Float(-99.)  # Parameter for chilling unit accumulation
        CSUMDB   = Float(-99.)  # Chilling unit sum for dormancy break

    class RateVariables(RatesTemplate):
        DTSUME = Float(-99.)  # increase in temperature sum for emergence
        DTSUM  = Float(-99.)  # increase in temperature sum
        DVR    = Float(-99.)  # development rate
        DCU    = Float(-99.)  # Daily chilling units

    class StateVariables(StatesTemplate):
        DVS    = Float(-99.)  # Development stage
        TSUME  = Float(-99.)  # Temperature sum for emergence state
        TSUM   = Float(-99.)  # Temperature sum state
        CSUM   = Float(-99.) # Chilling sum state
        # Based on the Elkhorn-Lorenz Grape Phenology Stage
        STAGE  = Enum(["endodorm", "ecodorm", "budbreak", "flowering", "verasion", "ripe"])
        STAGE_INT = Int(-.99) # Int of Stage
        DOP    = Instance(datetime.date) # Day of planting
        DOB    = Instance(datetime.date) # Day of bud break
        DOL    = Instance(datetime.date) # Day of Flowering
        DOV    = Instance(datetime.date) # Day of Verasion
        DOR    = Instance(datetime.date) # Day of Ripe
        DON    = Instance(datetime.date) # Day of Endodormancy
        DOC    = Instance(datetime.date) # Day of Ecodormancy
       
    def initialize(self, day:datetime.date, kiosk:VariableKiosk, parvalues:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE  instance
        :param parvalues: `ParameterProvider` object providing parameters as
                key/value pairs
        """
        self.params = self.Parameters(parvalues)
        self.kiosk = kiosk

        self._connect_signal(self._on_DORMANT, signal=signals.crop_dormant)
        # Define initial states
        DVS, DOP, STAGE = self._get_initial_stage(day)
        self.states = self.StateVariables(kiosk, 
                                          publish=["DVS", "TSUM", "TSUME", 
                                                   "STAGE", "DOP", "DOB","DON",
                                                   "DOL", "DOV", "DOR", "DOC" ],
                                          TSUM=0., TSUME=0., DVS=DVS, STAGE=STAGE, 
                                          DOP=DOP, CSUM=0.,DOB=None, DOL=None, DOV=None,
                                          DOR=None, DOC=None,DON=None, STAGE_INT=self._STAGE_VAL[STAGE])
        
        self.rates = self.RateVariables(kiosk, publish=["DTSUME", "DTSUM", "DVR", "DCU"])

    def _get_initial_stage(self, day:datetime.date):
        """Set the initial state of the crop given the start type
        """
        p = self.params

        # Define initial stage type (emergence/sowing) and fill the
        # respective day of sowing/emergence (DOS/DOE)
        if p.CROP_START_TYPE == "predorm":
            STAGE = "endodorm"
            DVS = p.DVSI
            DOP = day
        elif p.CROP_START_TYPE == "endodorm":
            STAGE = "ecodorm"
            DOP = day
            DVS = -0.1
        elif p.CROP_START_TYPE == "ecodorm":
            STAGE = "budbreak"
            DVS = p.DVSI
            DOP = day
        else:
            msg = "Unknown start type: %s. Are you using the corect Phenology \
                module (Calling the correct Gym Environment)?" % p.CROP_START_TYPE
            raise exc.PCSEError(msg) 
        return DVS, DOP, STAGE

    @prepare_rates
    def calc_rates(self, day, drv):
        """Calculates the rates for phenological development
        """
        p = self.params
        r = self.rates
        s = self.states

        # Day length sensitivity
        self._DAY_LENGTH =  daylength(day, drv.LAT)

        r.DTSUME = 0.
        r.DTSUM = 0.
        r.DVR = 0.
        r.DCU = 0.
        # Development rates
        if day.day == 1 and day.month == 8:
            self._CU_FLAG = True

        if s.STAGE == "endodorm":
            if self._CU_FLAG:
                r.DCU = p.Q10C**(-drv.TMAX/10) + p.Q10C**(-drv.TMIN/10)
            r.DTSUM =  max(0., daily_temp_units(drv, p.TBASEM, p.TEFFMX))
            r.DVR = r.DTSUM/(p.TSUM3-p.TSUM2)
        elif s.STAGE == "ecodorm":
            r.DTSUME = max(0., daily_temp_units(drv, p.TBASEM, p.TEFFMX))
            r.DVR = 0.1 * r.DTSUME/p.TSUMEM
        elif s.STAGE == "budbreak":
            r.DTSUM = max(0., daily_temp_units(drv, p.TBASEM, p.TEFFMX))
            r.DVR = r.DTSUM/(p.TSUM1)
        elif s.STAGE == "flowering":
            r.DTSUM =  max(0., daily_temp_units(drv, p.TBASEM, p.TEFFMX))
            r.DVR = r.DTSUM/(p.TSUM2-p.TSUM1)
            if self._CU_FLAG:
                r.DCU = p.Q10C**(-drv.TMAX/10) + p.Q10C**(-drv.TMIN/10)
        elif s.STAGE == "verasion":
            r.DTSUM =  max(0., daily_temp_units(drv, p.TBASEM, p.TEFFMX))
            r.DVR = r.DTSUM/(p.TSUM3-p.TSUM2)
            if self._CU_FLAG:
                r.DCU = p.Q10C**(-drv.TMAX/10) + p.Q10C**(-drv.TMIN/10)
        elif s.STAGE == "ripe":
            r.DTSUM = 0.
            r.DVR = 0.
            if self._CU_FLAG:
                r.DCU = p.Q10C**(-drv.TMAX/10) + p.Q10C**(-drv.TMIN/10)
            r.DTSUM =  max(0., daily_temp_units(drv, p.TBASEM, p.TEFFMX))
            r.DVR = r.DTSUM/(p.TSUM3-p.TSUM2)
        else:  # Problem: no stage defined
            msg = "Unrecognized STAGE defined in phenology submodule: %s."
            raise exc.PCSEError(msg, self.states.STAGE)

        msg = "Finished rate calculation for %s"
        self.logger.debug(msg % day)


        
    @prepare_states
    def integrate(self, day, delt=1.0):
        """Updates the state variable and checks for phenologic stages
        """

        p = self.params
        r = self.rates
        s = self.states

        # Integrate phenologic states
        s.TSUME += r.DTSUME
        s.DVS += r.DVR
        s.TSUM += r.DTSUM
        s.CSUM += r.DCU
        s.STAGE_INT = self._STAGE_VAL[s.STAGE]

        # Check if a new stage is reached
        if s.STAGE == "endodorm":
            if s.CSUM >= p.CSUMDB:
                self._next_stage(day)
                s.DVS = -0.1
                s.CSUM = 0
        elif s.STAGE == "ecodorm":
            if s.TSUME >= p.TSUMEM:
                self._next_stage(day)
                s.DVS = 0.
        elif s.STAGE == "budbreak":
            if s.DVS >= 1.0:
                self._next_stage(day)
                s.DVS = 1.0
        elif s.STAGE == "flowering":
            if s.DVS >= p.DVSM:
                self._next_stage(day)
                s.DVS = p.DVSM
        elif s.STAGE == "verasion":
            if s.DVS >= p.DVSEND:
                self._next_stage(day)
                s.DVS = p.DVSEND
            if self._DAY_LENGTH <= p.MLDORM:
                s.STAGE = "endodorm"
                s.DOC = day
        elif s.STAGE == "ripe":
            if self._DAY_LENGTH <= p.MLDORM:
                s.STAGE = "endodorm"
                s.DOC = day
        else:  # Problem: no stage defined
            msg = "Unrecognized STAGE defined in phenology submodule: %s."
            raise exc.PCSEError(msg, self.states.STAGE)
            
        msg = "Finished state integration for %s"
        self.logger.debug(msg % day)

    def _next_stage(self, day):
        """Moves states.STAGE to the next phenological stage"""
        s = self.states
        p = self.params

        current_STAGE = s.STAGE
        if s.STAGE == "endodorm":
            s.STAGE = "ecodorm"
            s.DON = day
            self._send_signal(signals.crop_dormant, day=day)
            self._CU_FLAG = False
        
        elif s.STAGE == "ecodorm":
            s.STAGE = "budbreak"
            s.DOB = day
                
        elif s.STAGE == "budbreak":
            s.STAGE = "flowering"
            s.DOL = day

        elif s.STAGE == "flowering":
            s.STAGE = "verasion"
            s.DOV = day
            
        elif s.STAGE == "verasion":
            s.STAGE = "ripe"
            s.DOR = day
            
        elif s.STAGE == "ripe":
            msg = "Cannot move to next phenology stage: maturity already reached!"
            raise exc.PCSEError(msg)

        else: # Problem no stage defined
            msg = "No STAGE defined in phenology submodule."
            raise exc.PCSEError(msg)
        
        msg = "Changed phenological stage '%s' to '%s' on %s"
        self.logger.info(msg % (current_STAGE, s.STAGE, day))

    def _on_DORMANT(self, day:datetime.date):
        """Handler for dormant signal. Reset all nonessential states and rates to 0
        """
        
        s = self.states
        r = self.rates

        s.TSUM  = 0
        s.TSUME = 0

        r.DTSUME = 0 
        r.DTSUM  = 0
        r.DVR    = 0
