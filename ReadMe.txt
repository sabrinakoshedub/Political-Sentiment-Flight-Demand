To run the models on different country pairs, different intervention dates, or with different target months, edit the configurations in the JSON files located in the 'Code' folder. To edit the configurations for the code that runs T100 data ('T100_functions.py', 'SARIMA_Functions.py', 'SARIMAX_Functions.py') , edit 'config_T100.json'. To edit the configurations for the code that runs booking data (BookingCurves-Functions), edit 'config_booking.json'. There are no configurations for 'Daily_Functions' as the original data is limited to just CA-US. These configurations will produce new graphs and tables for the selected parameters which will be found in the Tables/Figures folders after running. 


Available country pairs for T100 Data (ctry_fm-ctry_to): ['GB-US', 'NL-US', 'CH-US', 'IE-US', 'DE-US', 'IT-US', 'FR-US', 'BE-US', 'PT-US', 'NO-US', 'ES-US', 'SE-US', 'CA-US', 'AT-US', 'DK-US', 'FI-US']


Available country pairs for BOOKING Data: ['GB-US', 'NL-US', 'CH-US', 'IE-US', 'DE-US', 'IT-US', 'FR-US', 'BE-US', 'PT-US', 'NO-US', 'ES-US', 'SE-US', 'CA-US', 'AT-US', 'DK-US', 'FI-US']-US', 'AT-US', 'DK-US', 'FI-US']