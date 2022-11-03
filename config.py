Debug = True           # Shows information in terminal and plots raw data
UseSampleData = False   # Use the sample data file in /data/ folder
Channels = [            # A list of channels to import from mdf into the df
    # Speed Channels
    'Cadet_IP_Speed',
    'WhlRPM_RL',
    'WhlRPM_RR',
    'InshaftN',
    'ClushaftN',
    'MaishaftN',
    'OutshaftN',
    
    # Torque Channels
    'Cadet_IP_Torque',
    'Cadet_OP_Torque_1',
    'Cadet_OP_Torque_2',

    # Oil system channels
    'Cadet_Oil_flow',
    'Cadet_Oil_Pres',
    'Cadet_Oil_Temp',

    # Misc channels
    'CadetPhase',
    'GearEngd'
]

# Gear ratio data
Gears = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th"}
Ratios = {1: 12.803, 2: 9.267, 3: 7.058, 4: 5.581, 5: 4.562, 6: 3.878, 7: 3.435}
