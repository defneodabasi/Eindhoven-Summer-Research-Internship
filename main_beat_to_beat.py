# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:59:12 2023

@author: defne.odabasi
"""
import base64
import numpy as np
import pandas as pd
import xmltodict
import ecg_plot
import matplotlib.pyplot as plt
from scipy import signal
import scipy.stats as stats

import functions_VCG_SVD_ECG
import functions_beat_to_beat 
import ExtractPoints
from plotECG import decode_lead  
from plotECG import plotdata
import neurokit2 as nk
import seaborn as sns
#%%
folderpath = 'C:\\Users\\defne.odabasi\\Documents\\ECG\\ECG - Anonieme Data'
#filename = 'C02_ECG_9.xml'

filenames=[
# 'C02_ECG_1.xml',
# 'C02_ECG_10.xml',
 'C02_ECG_11.xml',
# 'C02_ECG_12.xml',
# 'C02_ECG_13.xml',
# 'C02_ECG_15.xml',
# 'C02_ECG_16.xml',
# 'C02_ECG_17.xml',
# 'C02_ECG_18.xml',
# 'C02_ECG_19.xml',
# 'C02_ECG_20.xml',
# 'C02_ECG_21.xml',
# 'C02_ECG_22.xml',
# 'C02_ECG_23.xml',
# 'C02_ECG_24.xml',
# 'C02_ECG_25.xml',
# 'C02_ECG_26.xml',
# 'C02_ECG_27.xml',
# 'C02_ECG_3.xml',
# 'C02_ECG_4.xml',    
# 'C02_ECG_5.xml',
# 'C02_ECG_7.xml',
# 'C02_ECG_8.xml', 
# 'C02_ECG_9.xml',
# 'C05_ECG_10.xml',
# 'C05_ECG_11.xml',
# 'C05_ECG_12.xml', 
# 'C05_ECG_14.xml',
# 'C05_ECG_15.xml',
# 'C05_ECG_4.xml',
# 'C05_ECG_6.xml',
# 'C05_ECG_8.xml',
# 'C05_ECG_9.xml',
# 'C06_ECG_1.xml',
# 'C06_ECG_10.xml',
# 'C06_ECG_11.xml',
# 'C06_ECG_12.xml',
# 'C06_ECG_13.xml',
# 'C06_ECG_14.xml',
# 'C06_ECG_3.xml',
# 'C06_ECG_4.xml',
# 'C06_ECG_5.xml',
# 'C06_ECG_6.xml',
# 'C06_ECG_7.xml',
# 'C06_ECG_8.xml',
# 'C06_ECG_9.xml',
# 'C07_ECG_10.xml',
# 'C07_ECG_11.xml',
# 'C07_ECG_2.xml',
# 'C07_ECG_3.xml',
# 'C07_ECG_5.xml',
# 'C07_ECG_4.xml',
# 'C07_ECG_8.xml',
# 'C07_ECG_9.xml',
# 'C10_ECG_1.xml',
# 'C10_ECG_10.xml',
# 'C10_ECG_11.xml',
# 'C10_ECG_12.xml',
# 'C10_ECG_13.xml',
# 'C10_ECG_14.xml',
# 'C10_ECG_15.xml',
# 'C10_ECG_18.xml',
# 'C10_ECG_2.xml',
# 'C10_ECG_20.xml',
# 'C10_ECG_21.xml',
# 'C10_ECG_23.xml',
# 'C10_ECG_22.xml',
# 'C10_ECG_3.xml',
# 'C10_ECG_4.xml',
# 'C10_ECG_5.xml',
# 'C10_ECG_6.xml',
# 'C10_ECG_8.xml',
# 'C11_ECG_10.xml',
# 'C11_ECG_11.xml',
# 'C11_ECG_12.xml',
# 'C11_ECG_13.xml',
# 'C11_ECG_14.xml',
# 'C11_ECG_15.xml',
# 'C11_ECG_16.xml',
# 'C11_ECG_18.xml',
# 'C11_ECG_19.xml',
# 'C11_ECG_20.xml',
# 'C11_ECG_3.xml',
# 'C11_ECG_4.xml',
# 'C11_ECG_5.xml',
# 'C11_ECG_6.xml',
# 'C11_ECG_7.xml',
# 'C11_ECG_8.xml',
# 'C14_ECG_1.xml',
# 'C14_ECG_10.xml',
# 'C14_ECG_11.xml',
# 'C14_ECG_12.xml',
# 'C14_ECG_13.xml',
# 'C14_ECG_14.xml',
# 'C14_ECG_15.xml',
# 'C14_ECG_16.xml',
# 'C14_ECG_17.xml', 
# 'C14_ECG_18.xml',
# 'C14_ECG_19.xml',
# 'C14_ECG_2.xml',
# 'C14_ECG_4.xml',
# 'C14_ECG_5.xml',
# 'C14_ECG_8.xml',
# 'C14_ECG_9.xml',
# 'C15_ECG_13.xml',
# 'C15_ECG_14.xml',
# 'C15_ECG_16.xml',
# 'C15_ECG_3.xml', 
# 'C15_ECG_4.xml',
# 'C15_ECG_5.xml',
# 'C15_ECG_6.xml',
# 'C15_ECG_7.xml',
# 'C15_ECG_9.xml',
# 'C18_ECG_11.xml',
# 'C18_ECG_13.xml',
# 'C18_ECG_14.xml',
# 'C18_ECG_15.xml',
# 'C18_ECG_16.xml',
# 'C18_ECG_17.xml',
# 'C18_ECG_19.xml', 
# 'C18_ECG_3.xml',
# 'C18_ECG_4.xml',
# 'C18_ECG_5.xml',
# 'C18_ECG_8.xml',
# 'C18_ECG_9.xml',
# 'C19_ECG_11.xml',
# 'C19_ECG_12.xml',
# 'C19_ECG_13.xml',
# 'C19_ECG_14.xml',
# 'C19_ECG_15.xml',
# 'C19_ECG_18.xml',
# 'C19_ECG_2.xml',
# 'C19_ECG_20.xml',
# 'C19_ECG_3.xml',
# 'C19_ECG_5.xml',
# 'C19_ECG_6.xml',
# 'C19_ECG_7.xml',
# 'C19_ECG_8.xml',
# 'C19_ECG_9.xml',
# 'C20_ECG_1.xml',
# 'C20_ECG_10.xml',
# 'C20_ECG_2.xml',
# 'C20_ECG_3.xml',
# 'C20_ECG_4.xml',
# 'C20_ECG_5.xml',
# 'C20_ECG_6.xml',
# 'C20_ECG_7.xml',
# 'C20_ECG_8.xml',
# 'C20_ECG_9.xml',
# 'C25_ECG_10.xml',
# 'C25_ECG_12.xml',
# 'C25_ECG_13.xml',
# 'C25_ECG_14.xml',
# 'C25_ECG_15.xml',
# 'C25_ECG_20.xml',
# 'C25_ECG_22.xml',
# 'C25_ECG_7.xml',
# 'C25_ECG_8.xml',
# 'C25_ECG_9.xml',
# 'C26_ECG_1.xml',
# 'C26_ECG_10.xml',
# 'C26_ECG_11.xml',
# 'C26_ECG_12.xml',
# 'C26_ECG_14.xml',
# 'C26_ECG_15.xml',
# 'C26_ECG_17.xml',
# 'C26_ECG_19.xml',
# 'C26_ECG_3.xml',
# 'C26_ECG_4.xml',
# 'C26_ECG_5.xml',
# 'C26_ECG_6.xml',
# 'C26_ECG_7.xml',
# 'C26_ECG_9.xml',
# 'C27_ECG_11.xml',
# 'C27_ECG_12.xml',
# 'C27_ECG_13.xml',
# 'C27_ECG_14.xml',
# 'C27_ECG_15.xml',
# 'C27_ECG_16.xml',
# 'C27_ECG_17.xml',
# 'C27_ECG_18.xml',
# 'C27_ECG_2.xml',
# 'C27_ECG_4.xml',
# 'C27_ECG_6.xml',
# 'C27_ECG_7.xml',
# 'C27_ECG_8.xml',
# 'C28_ECG_2.xml',
# 'C28_ECG_3.xml',
# 'C28_ECG_4.xml',
# 'C28_ECG_6.xml',
# 'C28_ECG_7.xml',
# 'C28_ECG_8.xml',
# 'C28_ECG_9.xml',
# 'C30_ECG_11.xml',
# 'C30_ECG_12.xml',
# 'C30_ECG_13.xml',
# 'C30_ECG_14.xml',
# 'C30_ECG_17.xml',
# 'C30_ECG_2.xml',
# 'C30_ECG_3.xml',
# 'C30_ECG_4.xml',
# 'C30_ECG_5.xml',
# 'C30_ECG_6.xml',
# 'C30_ECG_7.xml',
# 'C30_ECG_8.xml',
# 'C30_ECG_9.xml',
# 'C31_ECG_10.xml',
# 'C31_ECG_11.xml',
# 'C31_ECG_12.xml',
# 'C31_ECG_13.xml',
# 'C31_ECG_15.xml',
# 'C31_ECG_16.xml',
# 'C31_ECG_17.xml',
# 'C31_ECG_18.xml',
# 'C31_ECG_19.xml',
# 'C31_ECG_20.xml',
# 'C32_ECG_17.xml',
# 'C32_ECG_18.xml',
# 'C32_ECG_19.xml', 
# 'C32_ECG_20.xml',  
# 'C32_ECG_21.xml',
# 'C32_ECG_22.xml',
# 'C32_ECG_27.xml',
# 'C32_ECG_28.xml',
# 'C32_ECG_30.xml',
# 'C32_ECG_31.xml',
# 'C32_ECG_32.xml',
# 'C32_ECG_34.xml',
# 'C32_ECG_35.xml',
# 'C32_ECG_36.xml',
# 'C32_ECG_37.xml',
# 'C32_ECG_38.xml',
# 'C32_ECG_39.xml',
# 'C32_ECG_40.xml',
# 'C32_ECG_41.xml',
# 'C33_ECG_1.xml',
# 'C33_ECG_12.xml', 
# 'C33_ECG_13.xml',
# 'C33_ECG_14.xml',
# 'C33_ECG_15.xml',
# 'C33_ECG_16.xml',
# 'C33_ECG_17.xml',
# 'C33_ECG_18.xml',
# 'C33_ECG_19.xml',
# 'C33_ECG_2.xml',
# 'C33_ECG_20.xml',
# 'C33_ECG_22.xml',
# 'C33_ECG_23.xml',
# 'C33_ECG_24.xml',
# 'C33_ECG_25.xml',
# 'C33_ECG_26.xml',
# 'C33_ECG_27.xml',
# 'C33_ECG_28.xml',
# 'C33_ECG_3.xml',
# 'C33_ECG_35.xml',
# 'C33_ECG_6.xml',
# 'C33_ECG_7.xml',
# 'C33_ECG_8.xml',
# 'C34_ECG_1.xml',
# 'C34_ECG_10.xml',
# 'C34_ECG_11.xml',
# 'C34_ECG_12.xml',
# 'C34_ECG_14.xml',
# 'C34_ECG_15.xml',
# 'C34_ECG_16.xml',
# 'C34_ECG_17.xml',
# 'C34_ECG_18.xml',
# 'C34_ECG_2.xml', 
# 'C34_ECG_3.xml',
# 'C34_ECG_4.xml',
# 'C34_ECG_5.xml',
# 'C34_ECG_6.xml',
# 'C34_ECG_7.xml',
# 'C34_ECG_9.xml',
#'C35_ECG_1.xml',
# 'C35_ECG_10.xml',
# 'C35_ECG_11.xml',
# 'C35_ECG_12.xml',
# 'C35_ECG_13.xml',
# 'C35_ECG_14.xml',
# 'C35_ECG_15.xml',
# 'C35_ECG_16.xml',
# 'C35_ECG_2.xml',
# 'C35_ECG_21.xml',
# 'C35_ECG_22.xml',
# 'C35_ECG_3.xml',
# 'C35_ECG_5.xml',
# 'C35_ECG_6.xml',
# 'C35_ECG_7.xml',
# 'C35_ECG_8.xml',
# 'C35_ECG_9.xml',
# 'C37_ECG_1.xml',
# 'C37_ECG_10.xml',
# 'C37_ECG_11.xml',
# 'C37_ECG_12.xml',
# 'C37_ECG_13.xml',
# 'C37_ECG_14.xml',
# 'C37_ECG_15.xml',
# 'C37_ECG_16.xml',
# 'C37_ECG_17.xml',
# 'C37_ECG_18.xml',
# 'C37_ECG_19.xml',
# 'C37_ECG_2.xml',
# 'C37_ECG_20.xml',
# 'C37_ECG_21.xml',
# 'C37_ECG_22.xml',
# 'C37_ECG_24.xml',
# 'C37_ECG_3.xml',
# 'C37_ECG_4.xml',
# 'C37_ECG_7.xml',
# 'C37_ECG_8.xml',
# 'C37_ECG_9.xml',
# 'C39_ECG_1.xml',
# 'C39_ECG_10.xml',
# 'C39_ECG_11.xml',
# 'C39_ECG_12.xml',
# 'C39_ECG_15.xml',   
# 'C39_ECG_16.xml',   
# 'C39_ECG_17.xml',
# 'C39_ECG_18.xml',
# 'C39_ECG_19.xml',
# 'C39_ECG_2.xml',
# 'C39_ECG_20.xml',
# 'C39_ECG_21.xml',
# 'C39_ECG_22.xml',
# 'C39_ECG_23.xml',
# 'C39_ECG_3.xml',
# 'C39_ECG_5.xml',
# 'C39_ECG_6.xml',
# 'C39_ECG_7.xml',
# 'C39_ECG_8.xml',
##'C39_ECG_9.xml',
# 'C40_ECG_10.xml',
# 'C40_ECG_11.xml',
# 'C40_ECG_12.xml',
# 'C40_ECG_13.xml',
# 'C40_ECG_14.xml',
# 'C40_ECG_15.xml',
# 'C40_ECG_17.xml',
# 'C40_ECG_18.xml',
# 'C40_ECG_19.xml',
# 'C40_ECG_2.xml',
# 'C40_ECG_20.xml',
# 'C40_ECG_3.xml',
# 'C40_ECG_4.xml',
# 'C40_ECG_5.xml',
# 'C40_ECG_6.xml',
# 'C40_ECG_7.xml',
# 'C40_ECG_8.xml'
]

#%%
#filenames=[
# 'VT04_ECG_14.xml', 
# 'VT03_ECG_10.xml',
# 'VT03_ECG_11.xml',
# 'VT03_ECG_12.xml',
# 'VT03_ECG_13.xml',
# 'VT03_ECG_2.xml',
# 'VT03_ECG_3.xml',
# 'VT03_ECG_4.xml',
# 'VT03_ECG_5.xml',
# 'VT03_ECG_6.xml',
# 'VT03_ECG_7.xml',
# 'VT03_ECG_8.xml',
# 'VT03_ECG_9.xml',
# 'VT04_ECG_10.xml',
# 'VT04_ECG_11.xml',
# 'VT04_ECG_12.xml',
# 'VT04_ECG_13.xml',
# 'VT04_ECG_14.xml',
# 'VT04_ECG_15.xml',
# 'VT04_ECG_16.xml',
# 'VT04_ECG_18.xml',
# 'VT04_ECG_2.xml', 
# 'VT04_ECG_20.xml',
# 'VT04_ECG_22.xml',
# 'VT04_ECG_25.xml',
# 'VT04_ECG_3.xml',
# 'VT04_ECG_4.xml',
# 'VT04_ECG_5.xml',
# 'VT04_ECG_6.xml',
# 'VT04_ECG_7.xml',
# 'VT04_ECG_8.xml',
# 'VT07_ECG_10.xml',
# 'VT07_ECG_11.xml',
# 'VT07_ECG_13.xml',
# 'VT07_ECG_18.xml',
# 'VT07_ECG_19.xml',
# 'VT07_ECG_5.xml',
# 'VT07_ECG_6.xml',
# 'VT07_ECG_8.xml',
# 'VT08_ECG_1.xml',
# 'VT08_ECG_3.xml',
# 'VT08_ECG_4.xml',
# 'VT08_ECG_5.xml',
# 'VT08_ECG_6.xml',
# 'VT08_ECG_7.xml'
#] 

#working:

#not working:

#bunları sonradan dahil et:
#ValueError: max() arg is an empty sequence
#'C25_ECG_20.xml' çözüldü
#'C28_ECG_8.xml' çözüldü
#'C33_ECG_20.xml' çözüldü
#'C39_ECG_9.xml' at beat8 leadV1 has a very different beat at the end this is the problem. Burda son sıkıntılı beatleri göz ardı et buralarda hasta kıpırdamış olabilirmiş.
#'C35_ECG_1.xml' at beat5 leadV2 has a rise after indexToff,burda bu sıkıntılı  beat5 ve beat6i çıkar. Oynamış ölçüm yapan alet.

#'C33_ECG_2.xml' çözüldü
#indexQonset is 139 and indexQoffset is 162 but indexToff is 155. This is why it does not work. VCG size(156,3).

# 'C33_ECG_3.xml' çözüldü
#indexQonset is 130 and indexQoffset 153 but indexToff is 152. This is why it does not work. VCG size(153,3)

# 'C33_ECG_6.xml' çözüldü

# 'C40_ECG_3.xml' çözüldü

# 'C35_ECG_3.xml' indexToff value is way too before it actually forms.

#filenames=[
# 'C25_ECG_20.xml',
# 'C28_ECG_8.xml',
# 'C33_ECG_20.xml',
# 'C33_ECG_2.xml',
# 'C33_ECG_3.xml',
# 'C33_ECG_6.xml',
# 'C40_ECG_3.xml',
# 'C35_ECG_3.xml'
# 'C39_ECG_9.xml',
# 'C35_ECG_1.xml',
#]


#%%
#patient_ecg_data = pd.DataFrame(columns = ['Group','Patient', 'Recording', 'Beat' ,'TCRT', 'TMD', 'Date ratio (shifted)']) #recordings are collected
patient_ecg_data = pd.read_excel('C:\\Users\\defne.odabasi\\Documents\\VCG\\patient_ecg_data1.xlsx')
patient_anonymus = pd.read_excel('C:\\Users\\defne.odabasi\\Documents\\VCG\\ECG_Anonymized.xlsx')
std_patient_data = pd.read_excel('C:\\Users\\defne.odabasi\\Documents\\VCG\\std_patient_data.xlsx')

for filename in filenames:
    plt.close('all')
    
    patient = filename.split('_')[0] #from C02_ECG_18.xml, it takes 'C02' as the patient.
    recording = filename.split('_')[-1].split('.')[0] #from C02_ECG_18.xml, it takes '18' as the recording number.
    if patient[0] == 'C':
        group = 'C'
    else :
        group = 'VT'
    
    ind_=filename.find('_')         #text for _ is subfolder
    subfolder=filename[0:ind_]
    filepath=folderpath+'\\'+subfolder+'\\'+filename
    with open(filepath, "rb") as f:
        inDict = xmltodict.parse(f)
        f.close()
            
    # determine sample frequency         
    SampleFrequencyTime=int(inDict['RestingECG']['Waveform'][0]['SampleBase'])
    print(SampleFrequencyTime)
    #rhythm
    sub_dict=inDict['RestingECG']['Waveform'][1]['LeadData']
       
    ecg_data={}
    for ld in sub_dict: 
        ecg_data.update(decode_lead(ld))  #function to convert the leads
      
    df_ecg = pd.DataFrame(ecg_data)
    converted_ecg = functions_VCG_SVD_ECG.convert_ECG(df_ecg)
    
    #median
    median_ecg = {} 
    for ld in inDict['RestingECG']['Waveform'][0]['LeadData']: 
        median_ecg.update(decode_lead(ld)) #Decode the leads and add to empty dictionary
    median_ecg = pd.DataFrame(median_ecg)
    median_ecg_converted = functions_VCG_SVD_ECG.convert_ECG(median_ecg)
    
    index_Rpeak_median, _ =signal.find_peaks(median_ecg_converted['lead_I'], height= 0.6 * max(median_ecg_converted['lead_I']), distance = round(SampleFrequencyTime *0.200))
    
    #plotting Rpeak for median ecg
    SampleSize=len(median_ecg_converted['lead_I'])
    samples =list(range(SampleSize))
    t = np.array([sample / SampleFrequencyTime*1000 for sample in samples])   # transform to ms units
    
    plt.rcParams['font.size'] = 18
    
    plt.figure(figsize= (10,4), dpi = 150)
    plt.plot(t,median_ecg_converted['lead_I'],linewidth=2)
    plt.show()

    plt.grid(visible = True,which = 'both')

    Rpeak_median = index_Rpeak_median / SampleFrequencyTime * 1000
    plt.scatter(Rpeak_median,median_ecg_converted['lead_I'][index_Rpeak_median],color='r', linewidth=8, label='Rtop')
    
    plt.legend()
    plt.xlabel('time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Median ECG Voltage vs Time graph')
    print('done')
    #%%
    
    leads = ['lead_I', 'lead_II','lead_V1', 'lead_V2', 'lead_V3', 'lead_V4', 'lead_V5', 'lead_V6' ]
    peaks_df = pd.DataFrame(columns=leads)
    ma_ecg_df = pd.DataFrame(columns=leads)
    ma_qrs_df = pd.DataFrame(columns=leads)
    
    Rtops=ExtractPoints.ExtractRtops(filename,folderpath)
    Toff_median =ExtractPoints.ExtractTOffset(filename,folderpath)
    
    #index_Rtops
    index_Rtops = []
    for Rtop in Rtops:
        if SampleFrequencyTime == 500:
            index_Rtop = round(Rtop/2)
        elif SampleFrequencyTime == 250:
            index_Rtop = round(Rtop/4)
        elif SampleFrequencyTime == 240:
            index_Rtop = round(Rtop*240/1000)
        index_Rtops.append(index_Rtop)
    
    #index_Toff_median
    if SampleFrequencyTime == 500:
        index_Toff_median = round(Toff_median/2)
    elif SampleFrequencyTime == 250:
        index_Toff_median = round(Toff_median/4)
    elif SampleFrequencyTime == 240:
        index_Toff_median = round(Toff_median*240/1000)
        
    #since the Toffset is calculated with respect to the median ecg. 
    
    RT_median_diff = index_Toff_median -index_Rpeak_median[0]
    
    #nk library is used when the median indexToff point does not provide a good estimation
    try:
        _, waves_cwt = nk.ecg_delineate(converted_ecg['lead_I'], index_Rtops, sampling_rate = SampleFrequencyTime, method='cwt',show=False,show_type='bounds_T')
        RT_nk_list = []
        for i in range(len(index_Rtops)-1):
            if not np.isnan(waves_cwt['ECG_T_Offsets'][i]):
                RT_diff_nk = waves_cwt['ECG_T_Offsets'][i] - index_Rtops[i]
                RT_nk_list.append(RT_diff_nk)
        
        if RT_nk_list: #if the list is not empty
            RT_nk = int(np.nanmean(RT_nk_list))
            if (RT_nk > RT_median_diff + 40) or (RT_median_diff > RT_nk + 40): #if the median RT is not in a reasonable place
                RT_median_diff = RT_nk #use the value from the library
                print('indexToff is found from nk library')
    except ValueError:
        continue
    
    
    
    #%%
    #Segmentation according to pan tompkins vs segmentation according to the peak values of the ecg signal can be compared 
    
    lead = 'lead_I' #specify the lead
    converted_lead_ecg = converted_ecg[lead]
    
    ma_ecg = functions_beat_to_beat.pan_tompkins_qrs_detection(converted_lead_ecg,SampleFrequencyTime,plot=False)
    ma_qrs = functions_beat_to_beat.QRSpeaks(ma_ecg, SampleFrequencyTime)
    
    #%%
    
    #plotting the leads and finding out the indexes for the QRSpeak values within the rhythm.
    plt.figure() 
    position=[1,5,2,6,3,7,4,8]
    for i,col in enumerate(df_ecg.columns):
        y=df_ecg[col]
        (t, voltage)=plotdata(y, SampleFrequencyTime) #from bits to mV
        #plt.rcParams['font.size'] = 14
        
        ax=plt.subplot(4, 2, position[i])
        ax.set_ylim(-2,2)
        plt.plot(t,voltage)
        plt.show()
        
        #plt.tight_layout() #make sure plots do not overlap and figure space is effeciently used
        plt.subplots_adjust(wspace=0.2,hspace=0.5)
        plt.title(col,fontsize = 16)
        ax.set_xlabel('time (ms)', fontsize = 16) 
        ax.set_ylabel('voltage (mV)', fontsize = 16)
    
    #%%
    
    #plotting 'lead_I' alone
    y=df_ecg['lead_I']
    converted_lead_signal = functions_VCG_SVD_ECG.convert_ECG(df_ecg)['lead_I']
    
    (t, voltage)=plotdata(y, SampleFrequencyTime) #from bits to mV. I dont use voltage list
    t = np.array(t) #converting from list to numpy array
    
    plt.figure(figsize= (20,4), dpi = 100)
    plt.grid(True, linewidth = 0.5)
    
    plt.rcParams['font.size'] = 18
    
    plt.xticks(np.arange(0,t[-1] + 10,500))
    plt.plot(t, voltage, color = 'blue')
    plt.scatter(t[index_Rtops], converted_lead_signal[index_Rtops], color ='red', s = 50, marker = 'o', label = 'Rtops')
    for i in index_Rtops:
        plt.axvline(x=t[i], linestyle='--', linewidth = 2, color ='red')
    
    plt.legend()
    plt.xlabel('time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('lead_I Voltage vs Time graph with Rtop values')
    #%%
    #finding QRSpeaks for each lead.
    for i,col in enumerate(df_ecg.columns):
        y=df_ecg[col]
        (t, voltage)=plotdata(y, SampleFrequencyTime) #from bits to mV and to ms 
        
        # in case the QRSpeaks found for the specific lead does not have the same size value as the other leads.
        try: 
            peaks_df[col] = functions_beat_to_beat.QRSpeaks(voltage, SampleFrequencyTime) #finding the peak indicies in the signal and placing in a dataframe for further use
        except ValueError as e:
            #print(f'Cannot find QRSpeak from QRSpeaks function in lead {col}: {e}')
            
            
            # converted_lead_signal = functions_VCG_SVD_ECG.convert_ECG(df_ecg)[col]
            # Rpeaks = functions_beat_to_beat.QRSpeaks(converted_lead_signal,SampleFrequencyTime)
            # t = np.array(t) #converting from list to numpy array
            
            # plt.figure(figsize= (20,4), dpi = 100)
            # plt.grid(True, linewidth = 0.5)
            # plt.xticks(np.arange(0,t[-1] + 10,500))
            # plt.plot(t, converted_lead_signal, color = 'blue')
            # plt.scatter(t[Rpeaks], converted_lead_signal[Rpeaks], color ='red', s = 50, marker = '*')
            
            # plt.xlabel('time (ms)')
            # plt.ylabel('Voltage (mV)')
            # plt.title(f'R peak Locations of the {col} with error ')
            # plt.show()
            continue
    
    #%%
    #Segmentation according to pan tompkins vs segmentation according to the peak values of the ecg signal can be compared 
    
    for i,col in enumerate(ma_ecg_df.columns):
        
        # in case the QRSpeaks found for the specific lead does not have the same size value as the other leads.
        try: 
            ma_ecg_df[col] = functions_beat_to_beat.pan_tompkins_qrs_detection(converted_ecg[col],SampleFrequencyTime,plot=False)
            ma_qrs_df[col] = functions_beat_to_beat.pan_tompkins_peaks(ma_ecg_df[col], SampleFrequencyTime) #finding the peak indicies in the signal and placing in a dataframe for further use
        except ValueError as e:
            print(f'Error at finding QRSpeak from Pan-Tompkins in lead {col}: {e}')
            
            converted_lead_ecg = converted_ecg[col]
            Rpeaks_lead = functions_beat_to_beat.QRSpeaks(converted_lead_ecg,SampleFrequencyTime)
            
            ma_ecg_lead = ma_ecg_df[col]
            ma_qrs_lead = functions_beat_to_beat.pan_tompkins_peaks(ma_ecg_lead, SampleFrequencyTime)
    
            proportion = converted_lead_ecg[Rpeaks_lead].mean()/ma_ecg_lead[ma_qrs_lead].mean()
    
            plt.figure(figsize= (20,4), dpi = 100)
            plt.xticks(np.arange(0,len(converted_lead_ecg)+1,300))
            plt.plot(converted_lead_ecg, color = 'blue',label='Rhythm ECG')
            plt.plot(ma_ecg_lead*proportion, color = 'purple',label='Pan-Tompkins Outcome Signal')
            plt.scatter(Rpeaks_lead, converted_lead_ecg[Rpeaks_lead], color ='red', s = 50, marker = '*',label='QRS peaks')
            plt.scatter(ma_qrs_lead, ma_ecg_lead[ma_qrs_lead]*proportion, color ='purple', s = 50, marker = '*', label='Pan-Tompkins peaks')
            for x in ma_qrs_lead:
                plt.axvline(x=x,color='r',linestyle='--')
    
            plt.xlabel('Samples')
            plt.ylabel('Voltage (mV)')
            plt.title('Pan-Tompkins Peaks and Ecg Signal Peaks')
            plt.legend(loc='best')
            plt.show()
            continue
    
    _#%% 
    #since we found the peak values we can identify the number of beats
    beat_labels = functions_beat_to_beat.generate_beat_labels(index_Rtops)
    
    fiducial_list = ['index_t_TS', 'index_t_RS', 'indexToff', 'index_t_RS_a', 'index_t_RE_a', 'index_t_RE']
    parameter_list = ['TCRT','TMD']
    #rhythm_data = pd.DataFrame(data = [], index = parameters, columns = beat_labels)
    
    fiducial_points = pd.DataFrame(data = [],index = fiducial_list, columns = beat_labels)
    beat_to_beat_parameters = pd.DataFrame(data = [],index = parameter_list, columns = beat_labels)
    
    
    for beat in beat_labels:
        
        
        beat_num = int(beat.split('_')[1])
        
        start_index, end_index = functions_beat_to_beat.SegmentBeats(converted_ecg,index_Rtops,SampleFrequencyTime,beat_num, plot=False)
        
        beat_converted_ecg = converted_ecg[start_index:end_index].reset_index(drop=True) #by reset_index the index value is reset to 0.
        
        index_Toff = index_Rtops[beat_num] + RT_median_diff - start_index
        
        #check point
        if len(beat_converted_ecg) < index_Toff:
            if (beat_num != len(beat_labels) - 1):
                if (index_Rtops[beat_num+1]-20 <= index_Toff + start_index):
                    index_Toff = end_index - start_index -1
                    beat_converted_ecg = converted_ecg[start_index:end_index].reset_index(drop=True)
                    print('hayda Toffset yanlış bulunmuş '+beat)
                else:
                    beat_converted_ecg = converted_ecg[start_index: index_Rtops[beat_num] + RT_median_diff + 10].reset_index(drop=True) 
                    print('haydaaaa Toffsetten önce kesilmiş beat uzattık beati '+beat)
            elif beat_num == len(beat_labels) - 1:
                beat_converted_ecg = converted_ecg[start_index:end_index].reset_index(drop=True)
        elif len(beat_converted_ecg) ==  index_Toff: #there are couple cases where the sizes exatly matches
            index_Toff = index_Toff-1
        else :
            beat_converted_ecg = converted_ecg[start_index:end_index].reset_index(drop=True) #by reset_index the index value is reset to 0.
                    
        #%%
        #SVD_matrices function is used from functions_VCG_SVC_ECG
        LeadX_Kors_beat = functions_VCG_SVD_ECG.leads_kors(beat_converted_ecg)[0] #X lead after Kors transformation
        LeadY_Kors_beat = functions_VCG_SVD_ECG.leads_kors(beat_converted_ecg)[1] #Y lead after Kors transformation
        LeadZ_Kors_beat = functions_VCG_SVD_ECG.leads_kors(beat_converted_ecg)[2] #Z lead after Kors transformation
        VCG_Kors_beat = functions_VCG_SVD_ECG.leads_kors(beat_converted_ecg)[3] #Dataframe of X,Y, and Z leads after Kors transformation
        
        #%%
        #beat voltage values, energy_values_signal should be calculated
        S_matrix = functions_VCG_SVD_ECG.SVD_matrices(beat_converted_ecg)[0]
        S_matrix_k = functions_VCG_SVD_ECG.SVD_matrices(beat_converted_ecg)[1]
        sig = functions_VCG_SVD_ECG.SVD_matrices(beat_converted_ecg)[2]
        sig_k = functions_VCG_SVD_ECG.SVD_matrices(beat_converted_ecg)[3]
        u_k = functions_VCG_SVD_ECG.SVD_matrices(beat_converted_ecg)[4]
        
        #%%
        #Energy of ECG signal
        energy_max =  functions_VCG_SVD_ECG.SVD_energy(S_matrix_k, S_matrix)[0]
        energy_values_signal = functions_VCG_SVD_ECG.SVD_energy(S_matrix_k, S_matrix)[1]
        S1_vector =  functions_VCG_SVD_ECG.SVD_energy(S_matrix_k, S_matrix)[2]
        S2_vector =  functions_VCG_SVD_ECG.SVD_energy(S_matrix_k, S_matrix)[3]
        
        #%%
    
        #SVD_fucidials function is needed.
        #t_TP is the peak of the T_wave is defined as the point with the maximum energy after the endpoint of QRS complex
    
        #t_RE_a is the endpoint of the R peak in the time where the ECG energy is lower that %70.
        
        #The starting point t_TS of the T-wave is calculated from t_RE_a + 1/3 *(t_TP-t_RE_a).
        
        #sometimes the last beat does not appear fully
        if (beat_num == len(beat_labels) - 1) & (index_Rtops[beat_num] + RT_median_diff > end_index):
            TCRT = np.nan
            TMD = np.nan
            
        else:
            
            index_t_RE_a = functions_VCG_SVD_ECG.SVD_fiducials(energy_max, energy_values_signal, inDict, index_Toff)[0]
            index_t_RS_a = functions_VCG_SVD_ECG.SVD_fiducials(energy_max, energy_values_signal, inDict, index_Toff)[1]
            index_t_RE = functions_VCG_SVD_ECG.SVD_fiducials(energy_max, energy_values_signal, inDict, index_Toff)[2]
            index_t_RS = functions_VCG_SVD_ECG.SVD_fiducials(energy_max, energy_values_signal, inDict, index_Toff)[3]
            index_t_TP = functions_VCG_SVD_ECG.SVD_fiducials(energy_max, energy_values_signal, inDict, index_Toff)[4]
            index_t_TS = functions_VCG_SVD_ECG.SVD_fiducials(energy_max, energy_values_signal, inDict, index_Toff)[5]
            
            #time points
            t_TS = index_t_TS / SampleFrequencyTime * 1000
            t_RS = index_t_RS / SampleFrequencyTime * 1000
            Toff = index_Toff / SampleFrequencyTime * 1000
            t_RS_a = index_t_RS_a / SampleFrequencyTime * 1000
            t_RE_a = index_t_RE_a /SampleFrequencyTime * 1000
            t_RE = index_t_RE / SampleFrequencyTime * 1000
            t_TP = index_t_TP / SampleFrequencyTime * 1000
            
            fiducial_points[beat][0] = t_TS
            fiducial_points[beat][1] = t_RS
            fiducial_points[beat][2] = Toff
            fiducial_points[beat][3] = t_RS_a
            fiducial_points[beat][4] = t_RE_a
            fiducial_points[beat][5] = t_RE
    
        #%%
        #does not work prints: 
        #If there is no cell exceeding the threshold value, find nearest cell to threshold value 
        #print('no')
        ##if selected time point is located on index_t_TP
        #print('ECG median signal, T top cannot be detected')
        
        #ndatapoints_matrix_new, th, xx, yy = functions_VCG_SVD_ECG.SVD_signal_threshold(S1_vector, S2_vector, index_t_TS, plot_weigth='YES')
        #index_t_TE = functions_VCG_SVD_ECG.SVD_Toffset(ndatapoints_matrix_new, th, yy, xx, S1_vector, S2_vector, index_t_TP) #this is a different method for calculating t_TE
        
        
        #%%
        #   
            u_recon_2 = functions_VCG_SVD_ECG.SVD_signal_processing(S_matrix_k, energy_max, index_t_RS, index_t_RE, index_t_TS, index_Toff, u_k)[0]
            sig_recon_2 = functions_VCG_SVD_ECG.SVD_signal_processing(S_matrix_k, energy_max, index_t_RS, index_t_RE, index_t_TS, index_Toff, u_k)[1]
            DC_norm_S_T_matrix = functions_VCG_SVD_ECG.SVD_signal_processing(S_matrix_k, energy_max, index_t_RS, index_t_RE, index_t_TS, index_Toff, u_k)[2]
            DC_norm_S_QRS_matrix = functions_VCG_SVD_ECG.SVD_signal_processing(S_matrix_k, energy_max, index_t_RS, index_t_RE, index_t_TS, index_Toff, u_k)[3]
            
            try: 
                TCRT = functions_beat_to_beat.TCRT_parameter(energy_values_signal, index_t_TS, index_t_RS, index_Toff, DC_norm_S_T_matrix, DC_norm_S_QRS_matrix, index_t_RS_a, index_t_RE_a, 20)[0]
            except IndexError as e:
                TCRT = functions_beat_to_beat.TCRT_parameter(energy_values_signal, index_t_TS, index_t_RS, index_Toff, DC_norm_S_T_matrix, DC_norm_S_QRS_matrix, index_t_RS_a, index_t_RE_a, 0)[0]
                print(f'at beat:{beat_num} error {e} is solved by taking constant as 0')
            beat_to_beat_parameters[beat][0] = TCRT
        
        #%%
        #plotting these data_points on the figure
            # lead_ecg = beat_converted_ecg['lead_I']
            
            # SampleSize=len(lead_ecg)
            # samples =list(range(SampleSize))
            # t = np.array([sample / SampleFrequencyTime*1000 for sample in samples])   # transform to ms units
            
            # plt.figure()
            # plt.grid(visible = True,which = 'both')
            # plt.plot(t,lead_ecg,label = f'Beat at peak: {beat_num} for lead_I')
            
            # plt.axvline(x=t_RS,color='purple',linestyle='--',label='t_RS')
            # plt.scatter(t_RS, 0, color ='purple', s = 75, marker = '.')
        
            # plt.axvline(x=t_RS_a,color='blue',linestyle='--',label='t_RS_a')
            # plt.scatter(t_RS_a, 0, color ='blue', s = 75, marker = '.')
            
            # plt.axvline(x=t_RE_a,color='brown',linestyle='--',label='t_RE_a')
            # plt.scatter(t_RE_a,0, color ='brown', s = 75, marker = '.')
        
            # plt.axvline(x=t_RE,color='green',linestyle='--',label='t_RE')
            # plt.scatter(t_RE, 0, color ='green', s = 75, marker = '.')
            
            # plt.axvline(x=t_TS,color='darkgreen',linestyle='--',label='t_TS')
            # plt.scatter(t_TS, 0, color ='darkgreen', s = 75, marker = '.')
        
            # plt.axvline(x=t_TP,color='orange',linestyle='--',label='t_TP')
            # plt.scatter(t_TP, 0, color ='orange', s = 75, marker = '.')
            
            # if len(peaks_df['lead_I']) != len(index_Rtops):
            #     lead_peak_index = index_Rtops[beat_num] - start_index
            #     lead_peak = lead_peak_index / SampleFrequencyTime * 1000
            #     plt.axvline(x=lead_peak,color='r',linestyle='--',label='Approximate Peak')
            # else:
            #     if peaks_df['lead_I'][beat_num] + 30 < index_Rtops[beat_num]: #there possibly a wrong Rpeak calculation again
            #         lead_peak_index = index_Rtops[beat_num] - start_index
            #         lead_peak = lead_peak_index / SampleFrequencyTime * 1000
            #         plt.axvline(x=lead_peak,color='r',linestyle='--',label=' Approximate Peak')
                
            #     elif peaks_df['lead_I'][beat_num] -30 > index_Rtops[beat_num] :
            #         lead_peak_index = index_Rtops[beat_num] - start_index
            #         lead_peak = lead_peak_index / SampleFrequencyTime * 1000
            #         plt.axvline(x=lead_peak,color='r',linestyle='--',label=' Approximate Peak')
                
            #     else : 
            #         lead_peak_index = peaks_df['lead_I'][beat_num] - start_index
            #         lead_peak = lead_peak_index / SampleFrequencyTime * 1000
            #         plt.scatter(lead_peak, lead_ecg[lead_peak_index], color ='red', s = 50, marker = '*', label = 'Rpeak')
        
            # plt.scatter(Toff, lead_ecg[index_Toff], color = 'black', s=75, marker = '*' ,label = 'Toffset')
            
            
            # plt.xlabel('time (ms)')
            # plt.ylabel('Voltage (mV)')
            # plt.title(f'ECG Beat at {beat_num}')
            # plt.legend(loc='best',fontsize='x-small')
            # plt.show()
            
            #%%
            #plottiing for all the leads
            plt.figure() 
            position=[1,5,2,6,3,7,4,8]
            for i,col in enumerate(df_ecg.columns):
                lead_ecg = beat_converted_ecg[col]
                SampleSize=len(lead_ecg)
                samples =list(range(SampleSize))
                t = np.array([sample / SampleFrequencyTime*1000 for sample in samples])   # transform to ms units
                
                plt.rcParams['font.size'] = 16
                
                ax=plt.subplot(4, 2, position[i])
                plt.plot(t,lead_ecg)
                plt.show()

                plt.grid(visible = True,which = 'both')
                plt.plot(t,lead_ecg,label = f'Beat at peak: {beat_num} for lead_I')


                #plt.axvline(x=t_RS,color='purple',linestyle='--',label='t_RS')
                plt.scatter(t_RS, lead_ecg[index_t_RS], color ='purple', s = 80, marker = '.')
                plt.text(t_RS,lead_ecg[index_t_RS],'RS',color = 'purple')
        
                #plt.axvline(x=t_RS_a,color='blue',linestyle='--',label='t_RS_a')
                plt.scatter(t_RS_a, lead_ecg[index_t_RS_a], color ='blue', s = 80, marker = '.')
                plt.text(t_RS_a,lead_ecg[index_t_RS_a],'RS_a',color = 'blue')

                #plt.axvline(x=t_RE_a,color='brown',linestyle='--',label='t_RE_a')
                plt.scatter(t_RE_a,lead_ecg[index_t_RE_a], color ='brown', s = 80, marker = '.')
                plt.text(t_RE_a,lead_ecg[index_t_RE_a],'RE_a',color = 'brown')

                #plt.axvline(x=t_RE,color='green',linestyle='--',label='t_RE')
                plt.scatter(t_RE, lead_ecg[index_t_RE], color ='green', s = 80, marker = '.')
                plt.text(t_RE,lead_ecg[index_t_RE],'RE',color = 'green')

                #plt.axvline(x=t_TS,color='darkgreen',linestyle='--',label='t_TS')
                plt.scatter(t_TS, lead_ecg[index_t_TS], color ='darkgreen', s = 80, marker = '.')
                plt.text(t_TS, lead_ecg[index_t_TS],'TS',color = 'darkgreen')

                plt.axvline(x=t_TP,color='orange',linestyle='--',label='t_TP')
                plt.scatter(t_TP, lead_ecg[index_t_TP], color ='orange', s = 80, marker = '.')
                plt.text(t_TP,lead_ecg[index_t_TP],'TP',color = 'orange')

                lead_peak_index = index_Rtops[beat_num] - start_index
                lead_peak = lead_peak_index / SampleFrequencyTime * 1000
                plt.axvline(x=lead_peak,color='r',linestyle='--', linewidth=4, label='Approximate Peak')

                plt.scatter(Toff, lead_ecg[index_Toff], color = 'black', s=80, marker = '*' ,label = 'Toffset')
                plt.text(Toff,lead_ecg[index_Toff],'Toff',color = 'black')
                
                plt.subplots_adjust(wspace=0.2,hspace=0.5)
                plt.xlabel('time (ms)',fontsize=16)
                plt.ylabel('Voltage (mV)',fontsize=16)
                plt.title(f'ECG Beat at {beat_num} {col}',fontsize=16)
                #plt.suptitle('Example for Corrected Toffset Placement with neurokit2 lib', fontweight='bold')
    
        #%%TMD calculation
            x2 = S1_vector[index_t_TS:index_Toff]
            y2 = S2_vector[index_t_TS:index_Toff]
            TMD = functions_VCG_SVD_ECG.TMD_parameter(u_recon_2, sig_recon_2, x2, y2, plot='NO') #If you don't want to show plot, set plot='NO'
            
            beat_to_beat_parameters[beat][1] = TMD
            
        #%%
            date_shifted = pd.to_datetime(patient_anonymus[patient_anonymus['Filename'] == filename]['Date (shifted)'])
            max_mi_date = pd.to_datetime(patient_anonymus[patient_anonymus['Filename'] == filename]['Max MIDate (shifted)'])
            min_vt_date = pd.to_datetime(patient_anonymus[patient_anonymus['Filename'] == filename]['Min VTDate (shifted)'])
            
            days_vt_mi = (min_vt_date - max_mi_date).dt.days
            days_dateshifted_mi = (date_shifted - max_mi_date).dt.days
            date_ratio = round(float(days_dateshifted_mi/days_vt_mi),5)
            
            data_dict = {'Group' : group, 'Patient': patient, 'Recording': recording, 'Beat': beat_num, 'TCRT': TCRT, 'TMD': TMD, 'Date ratio (shifted)' : date_ratio }
            data = pd.DataFrame(data_dict, index=[0])
            patient_ecg_data = pd.concat([patient_ecg_data, data], ignore_index = True)
        
        #%%
            SavePath_VCG_plot = ''
            #functions_beat_to_beat.plot2d_and_3d(VCG_Kors_beat, filename, f"Beat:{beat_num}", SavePath_VCG_plot, save=0)
            
            origincalc = functions_VCG_SVD_ECG.origincalc(VCG_Kors_beat) #calculate the origin
            
            indexQonset_median = int(inDict['RestingECG']['RestingECGMeasurements']['QOnset'])
            indexQoffset_median = int(inDict['RestingECG']['RestingECGMeasurements']['QOffset'])
            
            if int(inDict['RestingECG']['Waveform'][0]['SampleBase'])==250:
                indexQonset_median = int(indexQonset_median/2)
                indexQoffset_median = int(indexQoffset_median/2)
            
            indexQonset_beat = index_Rtops[beat_num] - (index_Rpeak_median[0] - indexQonset_median) - start_index
            indexQoffset_beat = index_Rtops[beat_num] + (indexQoffset_median - index_Rpeak_median[0]) - start_index
            
            #to visualize but not necessary
            # Qonset_beat = indexQonset_beat / SampleFrequencyTime * 1000
            # Qoffset_beat = indexQoffset_beat / SampleFrequencyTime * 1000
            
            # plt.scatter(Qonset_beat, lead_ecg[indexQonset_beat], color = 'black', s=75, marker = '*' ,label = 'indexQonset_beat')
            # plt.scatter(Qoffset_beat, lead_ecg[indexQoffset_beat], color = 'black', s=75, marker = '*' ,label = 'indexQoffset_beat')
            
            if (len(VCG_Kors_beat) <= indexQoffset_beat) or (index_Toff < indexQoffset_beat):
                print('QRSpeak and Tpeak vectors could not be used in the VCGplot')
            else:
                QRSpeak = functions_beat_to_beat.distcalc(origincalc, inDict, VCG_Kors_beat, indexQonset_beat, indexQoffset_beat, index_Toff)[5] #Determine maxQRS vector index
                Tpeak = functions_beat_to_beat.distcalc(origincalc, inDict, VCG_Kors_beat, indexQonset_beat, indexQoffset_beat, index_Toff)[6] #Determine maxT vector index
                #Create plot of ECG signal
                 
                rpeakamp = functions_beat_to_beat.distcalc(origincalc, inDict, VCG_Kors_beat, indexQonset_beat, indexQoffset_beat, index_Toff)[0] #get the coordinates of the QRS peak
                tpeakamp = functions_beat_to_beat.distcalc(origincalc, inDict, VCG_Kors_beat, indexQonset_beat, indexQoffset_beat, index_Toff)[1] #get the coordinates of the T peak
                vecQRS = functions_VCG_SVD_ECG.findVec(origincalc, rpeakamp) #calculate the vector of the calculated origin to the QRS peak coordinates 
                vecT = functions_VCG_SVD_ECG.findVec(origincalc, tpeakamp) #calculate the vector of the calculated origin to the T peak coordinates
                anglepeakQRST = functions_VCG_SVD_ECG.calculateangle(vecQRS, vecT) #calculate the SPQRSTa
                
                #angleazimuthQRS = functions_VCG_SVD_ECG.angleazimuthelevationpeakQRScalc(vecQRS, vecT, origincalc, VCG_Kors_beat, radius=0.2, fig=None, colour='C6', angleplot='NO')[0]
                
                #plotting VCG's
                functions_beat_to_beat.plotprojection_QRSloop(filename, f"Beat:{beat_num}",vecQRS, vecT, origincalc, VCG_Kors_beat, beat_num, radius=0.2, fig=None, colour='C6')
    #%%
    TCRT_mean = beat_to_beat_parameters.transpose()['TCRT'].mean()
    TMD_mean = beat_to_beat_parameters.transpose()['TMD'].mean()

#%% Calculating the standard deviation per ecg recording for the patients
patient_TMD = np.array(patient_ecg_data[(patient_ecg_data['Patient'] == patient) & (patient_ecg_data['Recording'] == recording)]['TMD'])
std_patient_TMD = np.std(patient_TMD)

patient_TCRT = np.array(patient_ecg_data[(patient_ecg_data['Patient'] == patient) & (patient_ecg_data['Recording'] == recording)]['TCRT'])
std_patient_TCRT = np.std(patient_TCRT)

#%%

    
plt.figure()
sns.set(style='whitegrid',font_scale=2)
gfg1 = sns.swarmplot(patient_ecg_data[patient_ecg_data['Patient'] == patient], x = 'Date ratio (shifted)', y='TCRT',native_scale=True, dodge=True, size=6)
gfg1.set_ylim(-1,1)
plt.title(f'{patient} TCRT swarmplot for recordings')

plt.figure()
sns.set(style='whitegrid',font_scale=2)
gfg2 = sns.swarmplot(data = patient_ecg_data[patient_ecg_data['Patient'] == patient], x = 'Date ratio (shifted)', y='TMD',native_scale=True)
gfg2.set_ylim(0,120)
plt.title(f'{patient} TMD swarmplot for recordings')

plt.figure()
sns.set(style='whitegrid',font_scale=2)
gfg3 = sns.boxplot(data=patient_ecg_data, x='Patient', y= 'TCRT', showfliers=0,dodge=False)
gfg3.set_ylim(-1,1)
#sns.swarmplot(data=patient_ecg_data, x = 'Patient', y='TCRT',zorder = 1,color='black')
plt.title('TCRT boxplot for patients')

plt.figure()
sns.set(style='whitegrid',font_scale=2)
gfg4 = sns.boxplot(data=patient_ecg_data, x='Patient', y= 'TMD',showfliers=0,dodge=False)
gfg4.set_ylim(0,120)
#sns.swarmplot(data=patient_ecg_data, x = 'Patient', y='TMD',zorder = 1, color='black')
plt.title('TMD boxplot for patients')

plt.figure()
sns.set(style='whitegrid',font_scale=2)
gfg5 = sns.boxplot(data=patient_ecg_data, x='Group', y= 'TCRT', showfliers=0,hue='Group',dodge=False)
gfg5.set_ylim(-1,1)
plt.title('TCRT boxplot for patients')

plt.figure()
sns.set(style='whitegrid',font_scale=2)
gfg6 = sns.boxplot(data=patient_ecg_data, x='Group', y= 'TMD',showfliers=0,hue='Group',dodge=False)
gfg6.set_ylim(0,120)
plt.title('TMD boxplot for patients')
    
#boxplot for the std
plt.figure()
sns.set(style='whitegrid',font_scale=2)
sns.boxplot(data=std_patient_data, x='Patient', y= 'std of TCRT', showfliers=0, dodge=False)
plt.title('std of TCRT boxplot for patients')

plt.figure()
sns.set(style='whitegrid',font_scale=2)
sns.boxplot(data=std_patient_data, x='Patient', y= 'std of TMD',showfliers=0, dodge=False)
plt.title('std of TMD boxplot for patients')

#%% Simple Stats Test for TMD

parameter_list = ['TMD','TCRT'] 

for parameter in parameter_list:

    xVT=list(patient_ecg_data[patient_ecg_data['Group'] == 'VT'][parameter].dropna())
    xC=list(patient_ecg_data[patient_ecg_data['Group'] == 'C'][parameter].dropna())
    
    #Normality Check
    [shapirovaluet_VT, shapirovaluep_VT] = stats.shapiro(xVT) # p < 0.05 --> not normally distribured
    [shapirovaluet_C,  shapirovaluep_C]  = stats.shapiro(xC)  # p < 0.05 --> not normally distribured
    
    #Equal variance check
    [levenevaluet_VTC, levenevaluep_VTC] = stats.levene(xC,xVT) # p < 0.05 --> variances are unequal
    
    # Check if differences are statistically different                    
    if shapirovaluep_C<=0.05 or shapirovaluep_VT<=0.05:   # if not normally distributed
        [statisticvalue_VTC, pvalue_VTC] = stats.mannwhitneyu(xC, xVT) # p < 0.05 --> difference are statistically different
        test_name = 'Mann-Whitney U'
    elif levenevaluep_VTC<=0.05:                               # if normally distributed and equal variances
        [statisticvalue_VTC, pvalue_VTC] = stats.ttest_ind(xC, xVT, equal_var=False) # p < 0.05 --> difference are statistically different
        test_name = 'T-Test Unequal Variance'
    else:                                                           # if normally distributed and equal variances
        [statisticvalue_VTC, pvalue_VTC] = stats.ttest_ind(xC, xVT, equal_var=True) # p < 0.05 --> difference are statistically different
        test_name = 'T-Test Equal Variance'
    
    #Print useful information
    if pvalue_VTC<0.05:
        print(f'The differences in {parameter} are statistically different. Test done is '+ test_name + ' with p-value ' + str(pvalue_VTC))
    else:
        print(f'The differences in {parameter} are NOT statistically different. Test done is '+ test_name + ' with p-value ' + str(pvalue_VTC))

#%%
# plt.figure() 
# position=[1,5,2,6,3,7,4,8]
# for i,col in enumerate(df_ecg.columns):
#     lead_ecg = beat_converted_ecg[col]
#     SampleSize=len(lead_ecg)
#     samples =list(range(SampleSize))
#     t = np.array([sample / SampleFrequencyTime*1000 for sample in samples])   # transform to ms units

#     ax=plt.subplot(4, 2, position[i])
#     plt.plot(t,lead_ecg)
#     plt.show()

#     plt.grid(visible = True,which = 'both')
#     plt.plot(t,lead_ecg,label = f'Beat at peak: {beat_num} for lead_I')

#     plt.axvline(x=t_RS,color='purple',linestyle='--',label='t_RS')
#     plt.scatter(t_RS, 0, color ='purple', s = 75, marker = '.')

#     plt.axvline(x=t_RS_a,color='blue',linestyle='--',label='t_RS_a')
#     plt.scatter(t_RS_a, 0, color ='blue', s = 75, marker = '.')

#     plt.axvline(x=t_RE_a,color='brown',linestyle='--',label='t_RE_a')
#     plt.scatter(t_RE_a,0, color ='brown', s = 75, marker = '.')

#     plt.axvline(x=t_RE,color='green',linestyle='--',label='t_RE')
#     plt.scatter(t_RE, 0, color ='green', s = 75, marker = '.')

#     plt.axvline(x=t_TS,color='darkgreen',linestyle='--',label='t_TS')
#     plt.scatter(t_TS, 0, color ='darkgreen', s = 75, marker = '.')

#     plt.axvline(x=t_TP,color='orange',linestyle='--',label='t_TP')
#     plt.scatter(t_TP, 0, color ='orange', s = 75, marker = '.')

#     lead_peak_index = index_Rtops[beat_num] - start_index
#     lead_peak = lead_peak_index / SampleFrequencyTime * 1000
#     plt.axvline(x=lead_peak,color='r',linestyle='--',label='Approximate Peak')

#     plt.scatter(Toff, lead_ecg[index_Toff], color = 'black', s=75, marker = '*' ,label = 'Toffset')


#     # plt.xlabel('time (ms)')
#     # plt.ylabel('Voltage (mV)')
#     plt.title(f'ECG Beat at {beat_num} {col}')


    #plt.tight_layout() #make sure plots do not overlap and figure space is effeciently used

