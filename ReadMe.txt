The objective of this project was to use airline booking and flight data as a proxy to quantify the effect of this political sentiment change through statistical modeling. Included is an automated and scalable model that can be used to investigate
changes in flight demand (passengers flown/to fly) in or between any locations, for any reason. For in-depth details on the scope and modeling of this project see the intro and outline files. Included are my python code files and shell script for compilation. Most of the project analyzes data that are publicly available from the department of transportation website (T100). 

To run the models on different country pairs, different intervention dates, or with different target months, edit the configurations in the JSON files located in the 'Code' folder. To edit the configurations for the code that runs T100 data ('T100_functions.py', 'SARIMA_Functions.py', 'SARIMAX_Functions.py') , edit 'config_T100.json'. To edit the configurations for the code that runs booking data (BookingCurves-Functions), edit 'config_booking.json'. There are no configurations for 'Daily_Functions' as the original data is limited to just CA-US. These configurations will produce new graphs and tables for the selected parameters which will be found in the Tables/Figures folders after running. 


Available country pairs for T100 Data (ctry_fm-ctry_to): ['GB-US', 'NL-US', 'CH-US', 'IE-US', 'DE-US', 'IT-US', 'FR-US', 'BE-US', 'PT-US', 'NO-US', 'ES-US', 'SE-US', 'CA-US', 'AT-US', 'DK-US', 'FI-US']



Available country pairs for BOOKING Data: ['GB-US', 'NL-US', 'CH-US', 'IE-US', 'DE-US', 'IT-US', 'FR-US', 'BE-US', 'PT-US', 'NO-US', 'ES-US', 'SE-US', 'CA-US', 'AT-US', 'DK-US', 'FI-US']-US', 'AT-US', 'DK-US', 'FI-US']
