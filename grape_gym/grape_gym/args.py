"""Args configurations file includes: 
    - PCSE configuration file for WOFOST 8.0 Water and NPK limited Production
    - WOFOST Gym parameter configurations
"""

from dataclasses import dataclass, field

@dataclass
class WOFOST_Args:
    """Dataclass to be used for configuration WOFOST crop and soil model

    If left to default of None, values will be drawn from the .yaml files in 
    /env_config/crop_config/ and /env_config/soil_config/
    """

    # NPK Soil Dynamics params
    """Base soil supply of N available through mineralization kg/ha"""
    NSOILBASE: float = None   
    """Fraction of base soil N that comes available every day"""         
    NSOILBASE_FR: float = None 
    """Base soil supply of P available through mineralization kg/ha"""
    PSOILBASE: float = None   
    """Fraction of base soil P that comes available every day"""         
    PSOILBASE_FR: float = None 
    """Base soil supply of K available through mineralization kg/ha"""
    KSOILBASE: float = None   
    """Fraction of base soil K that comes available every day"""         
    KSOILBASE_FR: float = None 
    """Initial N available in the N pool (kg/ha)"""
    NAVAILI: float = None
    """Initial P available in the P pool (kg/ha)"""
    PAVAILI: float = None
    """Initial K available in the K pool (kg/ha)"""
    KAVAILI: float = None
    """Maximum N available in the N pool (kg/ha)"""
    NMAX: float = None
    """Maximum P available in the P pool (kg/ha)"""
    PMAX: float = None
    """Maximum K available in the K pool (kg/ha)"""
    KMAX: float = None
    """Background supply of N through atmospheric deposition (kg/ha/day)"""
    BG_N_SUPPLY: float = None
    """Background supply of P through atmospheric deposition (kg/ha/day)"""
    BG_P_SUPPLY: float = None
    """Background supply of K through atmospheric deposition (kg/ha/day)"""
    BG_K_SUPPLY: float = None
    """Maximum rate of surface N to subsoil"""
    RNSOILMAX: float = None
    """Maximum rate of surface P to subsoil"""
    RPSOILMAX: float = None     
    """Maximum rate of surface K to subsoil"""
    RKSOILMAX: float = None     
    """Relative rate of N absorption from surface to subsoil"""
    RNABSORPTION: float = None  
    """Relative rate of P absorption from surface to subsoil"""
    RPABSORPTION: float = None  
    """Relative rate of K absorption from surface to subsoil"""
    RKABSORPTION: float = None 
    """Relative rate of NPK runoff as a function of surface water runoff"""
    RNPKRUNOFF: float = None    

    # Waterbalance soil dynamics params
    """Field capacity of the soil"""
    SMFCF: float = None                  
    """Porosity of the soil"""
    SM0: float = None                                
    """Wilting point of the soil"""
    SMW: float = None                          
    """Soil critical air content (waterlogging)"""
    CRAIRC: float = None       
    """maximum percolation rate root zone (cm/day)"""
    SOPE: float = None    
    """maximum percolation rate subsoil (cm/day)"""
    KSUB: float = None                  
    """Soil rootable depth (cm)"""
    RDMSOL: float = None                            
    """Indicates whether non-infiltrating fraction of rain is a function of storm size (1) or not (0)"""
    IFUNRN: bool = None    
    """Maximum surface storage (cm)"""                               
    SSMAX: float = None                          
    """Initial surface storage (cm)"""
    SSI: float = None                   
    """Initial amount of water in total soil profile (cm)"""
    WAV: float = None 
    """Maximum fraction of rain not-infiltrating into the soil"""
    NOTINF: float = None
    """Initial maximum moisture content in initial rooting depth zone"""
    SMLIM: float = None  
    """CO2 in atmosphere (ppm)"""
    CO2: float = None  

    # Phenology Parameters
    """Temperature sum from sowing to emergence (C day)"""
    TSUMEM: float = None   
    """Base temperature for emergence (C)"""
    TBASEM: float = None
    """Maximum effective temperature for emergence (C day)"""
    TEFFMX: float = None
    """Temperature sum from emergence to anthesis (C day)"""
    TSUM1: float = None
    """Temperature sum from anthesis to maturity (C day)"""
    TSUM2: float = None
    """Temperature sum from maturity to death (C day)"""
    TSUM3: float = None
    """Initial development stage at emergence. Usually this is zero, but it can 
    be higher or crops that are transplanted (e.g. paddy rice)"""
    DVSI: float = None
    """Mature development stage"""
    DVSM: float = None
    """Final development stage"""
    DVSEND: float = None      
    """Daylength dormancy threshold"""
    MLDORM: float = None

@dataclass 
class Agro_Args:
    """Dataclass to be used for configuration WOFOST agromanagement file

    If left to default of None, values will be drawn from the .yaml files in 
    /env_config/agro_config
    """

    """Latitude for Weather Data"""
    latitude: float = None
    """Longitude for Weather Data"""
    longitude: float = None
    """Year for Weather Data"""
    year: int = None
    """Site Name"""
    site_name: str = None
    """Site Variation Name"""
    variation_name: str = None
    "Site Start Date in YYYY-MM-DD"
    site_start_date: str = None
    """Site End Date in YYYY-MM-DD"""
    site_end_date: str = None
    """Crop Name"""
    crop_name: str = None
    "Crop Variety Name"
    variety_name: str = None
    """Crop Start Date in YYYY-MM-DD"""
    crop_start_date: str = None
    """Crop Start type (emergence/sowing)"""
    crop_start_type: str = None
    """Crop End Date in YYYY-MM-DD"""
    crop_end_date: str = None
    """Crop end type (harvest/maturity)"""
    crop_end_type: str = None
    """Max duration of crop growth"""
    max_duration: str = None

@dataclass
class NPK_Args:
    """Arguments for the WOFOST Gym environment
    """

    """Parameters for the WOFOST8 model"""
    wf_args: WOFOST_Args

    """Parameters for Agromanangement file"""
    ag_args: Agro_Args

    """Environment seed"""
    seed: int = 0
    
    """Output Variables"""
    """See env_config/README.md for more information"""
    output_vars: list = field(default_factory = lambda: ['DVS'])
    """Weather Variables"""
    weather_vars: list = field(default_factory = lambda: ['IRRAD', 'TEMP', 'RAIN'])

    """Intervention Interval"""
    intvn_interval: int = 1
    """Weather Forecast length in days (min 1)"""
    forecast_length: int = 1
    forecast_noise: list = field(default_factory = lambda: [0, 0.2])
    """Number of NPK Fertilization Actions"""
    """Total number of actions available will be 3*num_fert + num_irrig"""
    num_fert: int = 4
    """Number of Irrgiation Actions"""
    num_irrig: int = 4
    """Harvest Effiency in range (0,1)"""
    harvest_effec: float = 1.0
    """Irrigation Effiency in range (0,1)"""
    irrig_effec: float = 0.7

    """Coefficient for Nitrogen Recovery after fertilization"""
    n_recovery: float = 0.7
    """Coefficient for Phosphorous Recovery after fertilization"""
    p_recovery: float = 0.7
    """Coefficient for Potassium Recovery after fertilization"""
    k_recovery: float = 0.7
    """Amount of fertilizer coefficient in kg/ha"""
    fert_amount: float = 2
    """Amount of water coefficient in cm/water"""
    irrig_amount: float  = 0.5

    """Flag for resetting to random year"""
    random_reset: bool = False