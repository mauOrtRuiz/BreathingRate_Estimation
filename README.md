First we extract the data fields from the .json

Next we implement the following basic processing, and we base on https://pmc.ncbi.nlm.nih.gov/articles/PMC8402314/pdf/sensors-21-05651.pdf 


1.-As a basic preprocessing the RR signal is resampled to 6 Hz
2..- We apply a BP filter. For this task a 4th order Butterworth filter with frequencies from 0.2 Hz to 1.2 Hz was designed and applied to the resampled data.
Here we can see both signals bafore and after filtering:

![image](https://github.com/user-attachments/assets/a914d29c-1485-4fa7-9f44-a9da83fb1c30)

