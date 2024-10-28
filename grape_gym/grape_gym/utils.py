
"""Utils file for making model configurations and setting parameters from arguments
"""
import gymnasium as gym
from datetime import datetime
from grape_gym.args import WOFOST_Args, Agro_Args


from grape_model.soil.soil_wrappers import BaseSoilModuleWrapper, SoilModuleWrapper_LNPKW
from grape_model.crop.phenology import Grape_Phenology
from grape_model.agromanager import BaseAgroManager, AgroManagerPerennial


def make_config(soil: BaseSoilModuleWrapper=SoilModuleWrapper_LNPKW, crop: Grape_Phenology=Grape_Phenology, \
                agro: BaseAgroManager=AgroManagerPerennial):
    """Makes the configuration dictionary to be used to set various values of
    the model.
    
    Further modified in the WOFOST Gym delcaration.
    
    Args:
        None
    """

    # Module to be used for water balance
    SOIL = soil

    # Module to be used for the crop simulation itself
    CROP = crop

    # Module to use for AgroManagement actions
    AGROMANAGEMENT = agro

    # interval for OUTPUT signals, either "daily"|"dekadal"|"monthly"
    # For daily output you change the number of days between successive
    # outputs using OUTPUT_INTERVAL_DAYS. For dekadal and monthly
    # output this is ignored.
    OUTPUT_INTERVAL = "daily"
    OUTPUT_INTERVAL_DAYS = 1
    # Weekday: Monday is 0 and Sunday is 6
    OUTPUT_WEEKDAY = 0

    # variables to save at OUTPUT signals
    # Set to an empty list if you do not want any OUTPUT
    OUTPUT_VARS = [ 
   
        # PHENOLOGY STATES
        "DVS", "TSUM", "TSUME", "STAGE", "DOP", "DOC", "DON", "DOB", "DOF", "DOV", "DOR", "DOL",
        # PHENOLOGY RATES
        "DTSUME", "DTSUM", "DVR", "DCU",
        ]

    # Summary variables to save at CROP_FINISH signals
    # Set to an empty list if you do not want any SUMMARY_OUTPUT
    SUMMARY_OUTPUT_VARS = OUTPUT_VARS

    # Summary variables to save at TERMINATE signals
    # Set to an empty list if you do not want any TERMINAL_OUTPUT
    TERMINAL_OUTPUT_VARS = OUTPUT_VARS

    return {'SOIL': SOIL, 'CROP': CROP, 'AGROMANAGEMENT': AGROMANAGEMENT, 'OUTPUT_INTERVAL': OUTPUT_INTERVAL, \
            'OUTPUT_INTERVAL_DAYS':OUTPUT_INTERVAL_DAYS, 'OUTPUT_WEEKDAY': OUTPUT_WEEKDAY, \
                'OUTPUT_VARS': OUTPUT_VARS, 'SUMMARY_OUTPUT_VARS': SUMMARY_OUTPUT_VARS, \
                    'TERMINAL_OUTPUT_VARS': TERMINAL_OUTPUT_VARS}

def set_params(env: gym.Env, args: WOFOST_Args):
    """Sets editable WOFOST Model parameters by overriding the value
    in the configuration .yaml file
    
    Args:
        args - WOFOST_Args dataclass
    """

    # NPK Soil Dynamics params
    """Base soil supply of N available through mineralization kg/ha"""
    if args.NSOILBASE is not None:
        env.parameterprovider.set_override("NSOILBASE", args.NSOILBASE, check=False)  
    """Fraction of base soil N that comes available every day"""    
    if args.NSOILBASE_FR is not None:     
        env.parameterprovider.set_override("NSOILBASE_FR", args.NSOILBASE_FR, check=False)  
    """Base soil supply of P available through mineralization kg/ha"""
    if args.PSOILBASE is not None:
        env.parameterprovider.set_override("PSOILBASE", args.PSOILBASE, check=False)
    """Fraction of base soil P that comes available every day"""         
    if args.PSOILBASE_FR is not None:
        env.parameterprovider.set_override("PSOILBASE_FR", args.PSOILBASE_FR, check=False)
    """Base soil supply of K available through mineralization kg/ha"""
    if args.KSOILBASE is not None:
        env.parameterprovider.set_override("KSOILBASE", args.KSOILBASE, check=False)
    """Fraction of base soil K that comes available every day""" 
    if args.KSOILBASE_FR is not None:        
        env.parameterprovider.set_override("KSOILBASE_FR", args.KSOILBASE_FR, check=False)
    """Initial N available in the N pool (kg/ha)"""
    if args.NAVAILI is not None:
        env.parameterprovider.set_override("NAVAILI", args.NAVAILI, check=False)
    """Initial P available in the P pool (kg/ha)"""
    if args.PAVAILI is not None:
        env.parameterprovider.set_override("PAVAILI", args.PAVAILI, check=False)
    """Initial K available in the K pool (kg/ha)"""
    if args.KAVAILI is not None:
        env.parameterprovider.set_override("KAVAILI", args.KAVAILI, check=False)
    """Maximum N available in the N pool (kg/ha)"""
    if args.NMAX is not None:
        env.parameterprovider.set_override("NMAX", args.NMAX, check=False)
    """Maximum P available in the P pool (kg/ha)"""
    if args.PMAX is not None:
        env.parameterprovider.set_override("PMAX", args.PMAX, check=False)
    """Maximum K available in the K pool (kg/ha)"""
    if args.KMAX is not None:
        env.parameterprovider.set_override("KMAX", args.KMAX, check=False)
    """Background supply of N through atmospheric deposition (kg/ha/day)"""
    if args.BG_N_SUPPLY is not None:
        env.parameterprovider.set_override("BG_N_SUPPLY", args.BG_N_SUPPLY, check=False)
    """Background supply of P through atmospheric deposition (kg/ha/day)"""
    if args.BG_P_SUPPLY is not None:
        env.parameterprovider.set_override("BG_P_SUPPLY", args.BG_P_SUPPLY, check=False)
    """Background supply of K through atmospheric deposition (kg/ha/day)"""
    if args.BG_K_SUPPLY is not None:
        env.parameterprovider.set_override("BG_K_SUPPLY", args.BG_K_SUPPLY, check=False)
    """Maximum rate of surface N to subsoil"""
    if args.RNSOILMAX is not None:
        env.parameterprovider.set_override("RNSOILMAX", args.RNSOILMAX, check=False)
    """Maximum rate of surface P to subsoil"""
    if args.RPSOILMAX is not None:
        env.parameterprovider.set_override("RPSOILMAX", args.RPSOILMAX, check=False)  
    """Maximum rate of surface K to subsoil"""
    if args.RKSOILMAX is not None:
        env.parameterprovider.set_override("RKSOILMAX", args.RKSOILMAX, check=False)     
    """Relative rate of N absorption from surface to subsoil"""
    if args.RNABSORPTION is not None:
        env.parameterprovider.set_override("RNABSORPTION", args.RNABSORPTION, check=False)     
    """Relative rate of P absorption from surface to subsoil"""
    if args.RPABSORPTION is not None:
        env.parameterprovider.set_override("RPABSORPTION", args.RPABSORPTION, check=False)     
    """Relative rate of K absorption from surface to subsoil"""
    if args.RKABSORPTION is not None:
        env.parameterprovider.set_override("RKABSORPTION", args.RKABSORPTION, check=False) 
    """Relative rate of NPK runoff as a function of surface water runoff"""
    if args.RNPKRUNOFF is not None:
         env.parameterprovider.set_override("RNPKRUNOFF", args.RNPKRUNOFF, check=False) 
    # Waterbalance soil dynamics params
    """Field capacity of the soil"""
    if args.SMFCF is not None:
        env.parameterprovider.set_override("SMFCF", args.SMFCF, check=False)             
    """Porosity of the soil"""
    if args.SM0 is not None:
        env.parameterprovider.set_override("SM0", args.SM0, check=False)                            
    """Wilting point of the soil"""
    if args.SMW is not None:    
        env.parameterprovider.set_override("SMW", args.SMW, check=False)                  
    """Soil critical air content (waterlogging)"""
    if args.CRAIRC is not None:
        env.parameterprovider.set_override("CRAIRC", args.CRAIRC, check=False)
    """maximum percolation rate root zone (cm/day)"""
    if args.SOPE is not None:
        env.parameterprovider.set_override("SOPE", args.SOPE, check=False)
    """maximum percolation rate subsoil (cm/day)"""
    if args.KSUB is not None:
        env.parameterprovider.set_override("KSUB", args.KSUB, check=False)
    """Soil rootable depth (cm)"""
    if args.RDMSOL is not None:
        env.parameterprovider.set_override("RDMSOL", args.RDMSOL, check=False)                     
    """Indicates whether non-infiltrating fraction of rain is a function of storm size (1) or not (0)"""
    if args.IFUNRN is not None:
        env.parameterprovider.set_override("IFUNRN", args.IFUNRN, check=False)
    """Maximum surface storage (cm)"""                               
    if args.SSMAX is not None:
        env.parameterprovider.set_override("SSMAX", args.SSMAX, check=False)               
    """Initial surface storage (cm)"""
    if args.SSI is not None:
        env.parameterprovider.set_override("SSI", args.SSI, check=False)   
    """Initial amount of water in total soil profile (cm)"""
    if args.WAV is not None:
        env.parameterprovider.set_override("WAV", args.WAV, check=False)
    """Maximum fraction of rain not-infiltrating into the soil"""
    if args.NOTINF is not None:   
        env.parameterprovider.set_override("NOTINF", args.NOTINF, check=False)
    """Initial maximum moisture content in initial rooting depth zone"""
    if args.SMLIM is not None:
        env.parameterprovider.set_override("SMLIM", args.SMLIM, check=False)

    # Phenology Parameters
    """Temperature sum from sowing to emergence (C day)"""
    if args.TSUMEM is not None:
        env.parameterprovider.set_override("TSUMEM", args.TSUMEM, check=False)
    """Base temperature for emergence (C)"""
    if args.TBASEM is not None:
        env.parameterprovider.set_override("TBASEM", args.TBASEM, check=False)
    """Maximum effective temperature for emergence (C day)"""
    if args.TEFFMX is not None:
        env.parameterprovider.set_override("TEFFMX", args.TEFFMX, check=False)
    """Temperature sum from emergence to anthesis (C day)"""
    if args.TSUM1 is not None:
        env.parameterprovider.set_override("TSUM1", args.TSUM1, check=False)
    """Temperature sum from anthesis to maturity (C day)"""
    if args.TSUM2 is not None:
        env.parameterprovider.set_override("TSUM2", args.TSUM2, check=False)
    """Initial development stage at emergence. Usually this is zero, but it can 
    be higher or crops that are transplanted (e.g. paddy rice)"""
    if args.DVSI is not None:
        env.parameterprovider.set_override("DVSI", args.DVSI, check=False)
    """Mature Development Stage"""
    if args.DVSM is not None:
        env.parameterprovider.set_override("DVSM", args.DVSM, check=False)
    """Final development stage"""
    if args.DVSEND is not None:
        env.parameterprovider.set_override("DVSEND", args.DVSEND, check=False)
    """Daylength dormancy threshold"""
    if args.MLDORM is not None:
        env.parameterprovider.set_override("MLDORM", args.MLDORM, check=False)  

def set_agro_params(agromanagement: dict, args: Agro_Args):
    """Sets editable Agromanagement parameters by modifying the agromanagement
    dictionary before being passed to the AgroManager Module
    
    Args:
        args - Agro_Args dataclass
    """
    if args.latitude is not None:
        agromanagement['SiteCalendar']['latitude'] = args.latitude
    if args.longitude is not None: 
        agromanagement['SiteCalendar']['longitude'] = args.longitude
    if args.year is not None:
        agromanagement['SiteCalendar']['year'] = args.year
    if args.site_name is not None:
        agromanagement['SiteCalendar']['site_name'] = args.site_name
    if args.variation_name is not None:
        agromanagement['SiteCalendar']['variation_name'] = args.variation_name
    if args.site_start_date is not None:
        agromanagement['SiteCalendar']['site_start_date'] = datetime.strptime(args.site_start_date, '%Y-%m-%d').date()
    if args.site_end_date is not None:
        agromanagement['SiteCalendar']['site_end_date'] = datetime.strptime(args.site_end_date, '%Y-%m-%d').date()
    if args.crop_name is not None:
        agromanagement['CropCalendar']['crop_name'] = args.crop_name
    if args.variety_name is not None:
        agromanagement['CropCalendar']['variety_name'] = args.variety_name
    if args.crop_start_date is not None:
        agromanagement['CropCalendar']['crop_start_date'] = datetime.strptime(args.crop_start_date, '%Y-%m-%d').date()
    if args.crop_start_type is not None:
        agromanagement['CropCalendar']['crop_start_type'] = args.crop_start_type
    if args.crop_end_date is not None:
        agromanagement['CropCalendar']['crop_end_date'] = datetime.strptime(args.crop_end_date, '%Y-%m-%d').date()
    if args.crop_end_type is not None:
        agromanagement['CropCalendar']['crop_end_type'] = args.crop_end_type
    if args.max_duration is not None:
        agromanagement['CropCalendar']['max_duration'] = args.max_duration

    return agromanagement