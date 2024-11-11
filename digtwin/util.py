
"""
Utils file for making model configurations and setting parameters from arguments
"""
from datetime import datetime

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
        "DVS", "TSUM", "TSUME", "STAGE", "DOP", "DOC", "DON", "DOB", "DOF", "DOV", "DOR", "DOL", "STAGE_INT",
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