# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:35:04 2023

@author: defne.odabasi
"""

#%%
#%%Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import signal
from tabulate import tabulate
from math import sin, cos
import xmltodict
from plotECG import decode_lead  
import neurokit2 as nk
#%%
def convert_ECG(ecg_signal):
    '''
    From Margot's Code: Converts the input ECG data from bits to millivolts (mV).

    Parameters
    ----------
    ecg_signal : DataFrame
        Input signal (ECG) in bits.
    
    Returns
    -------
    converted_signal : DataFrame
        Converted signal values (ECG) in millivolts (mV).

    '''
    converted_signal = ecg_signal*(4.88/1000)
    return converted_signal

#%%
def pan_tompkins_qrs_detection(converted_signal,SampleFrequencyTime,plot=True):
    
    '''
    In this function, R peaks will be detected so that sampling of the ecg signal will be conducted smoothly
    Using this function to detect R_peak_index in the QRS complex
    
    Parameters
    ----------
    Converted_signal : DataFrame
        Input Signal in mV
    SampleFrequencyTime : int
    
    Returns
    -------
    
    
    '''
    SampleSize=len(converted_signal)
    samples =list(range(SampleSize))
    t_1 = [sample / SampleFrequencyTime*1000 for sample in samples]
    
    
    #the signal is already a clean filtered signal so bandpass is not included
    #plotting the rhythm signal
    if plot==True:
        plt.figure(figsize= (20,4),dpi =100)
        plt.xticks(np.arange(0, t_1[-1]+1,500))
        #plt.plot(converted_signal[32:len(converted_signal)-2])
        plt.plot(t_1,converted_signal)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title('Rhythm Signal')
    
    #taking the derivative for the signal
    #this step is taken so that we can obtain slope information for the signal. Thus the rate of change of the input is obtained.
    
    diff_ecg = np.diff(converted_signal)
    signal_squared = np.power(diff_ecg,2)
    
    #squaring the signal:
    SampleSize=len(signal_squared)
    samples =list(range(SampleSize))
    t_2 = [sample / SampleFrequencyTime*1000 for sample in samples]
        
    #plotting the result for squared signal
    if plot==True: 
        plt.figure(figsize= (20,4),dpi =100)
        #plt.xticks(np.arange(0, len(signal_squared)+1,200))
        plt.xticks(np.arange(0, t_2[-1]+1,500))
        #plt.plot(signal_squared[32:len(signal_squared)-2])
        plt.plot(t_2,signal_squared)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.title('Differentiated and Squared Signal')
    
    
    #moving_window_integration:
    
    window_size = round(0.150 * SampleFrequencyTime)
    ma_ecg = np.convolve(signal_squared, np.ones(window_size)/window_size, mode='same')
    
    SampleSize=len(ma_ecg)
    samples =list(range(SampleSize))
    t_3 = [sample / SampleFrequencyTime*1000 for sample in samples]
    
    if plot==True:
        #plotting the result for moving window average
        plt.figure(figsize= (20,4),dpi =100)
        #plt.xticks(np.arange(0, len(ma_ecg)+1,200))
        plt.xticks(np.arange(0, t_3[-1]+1,500))
        #plt.plot(ma_ecg[100:len(ma_ecg)-2])
        plt.plot(t_3,ma_ecg)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.title('Moving Window Integrated Signal')
        
    ma_ecg_peaks = pan_tompkins_peaks(ma_ecg, SampleFrequencyTime)
    
    if plot==True:
        fig,axes = plt.subplots(nrows=2,ncols=2, dpi = 100)
        
        axes[0][0].plot(t_1,converted_signal)
        axes[0][0].set_ylabel('Voltage (mV)')
        axes[0][0].set_title('Rhythm Signal')
        
        axes[0][1].plot(t_2,signal_squared)
        axes[0][1].set_ylabel('Amplitude')
        axes[0][1].set_title('Differentiated and Squared Signal')
        
        axes[1][0].plot(t_3,ma_ecg)
        axes[1][0].set_ylabel('Amplitude')
        axes[1][0].set_title('Moving Window Integrated Signal')
        
        axes[1][1].plot(t_3,ma_ecg*300,label='Integrated Signal')
        axes[1][1].plot(t_1,converted_signal,label='Rhythm Signal')
        axes[1][1].set_ylabel('Amplitude')
        axes[1][1].set_title('QRS peaks in Moving Window Integrated Signal')
        for x in ma_ecg_peaks:
             axes[1][1].axvline(x=t_3[x],color='r',linestyle='--')
             axes[1][1].scatter(t_3[x], ma_ecg[x] *300, color ='red', s = 50, marker = '*')
        
        axes[1][1].legend(bbox_to_anchor=(1.2,1),loc='upper right')
        
             
        fig.suptitle('Pan-Tompkins Algorithm Implementation Figures',fontweight='bold')

    return ma_ecg
#%%
#R peaks to be determined
def pan_tompkins_peaks(ma_ecg_lead,SampleFrequencyTime):
    '''
    This function finds the QRSpeak values for a specific lead in the signal.
    Parameters
    ----------
    converted_signal : DataFrame
        (converted_signal = convert_ECG(df_ecg)[lead])
    SampleFrequencyTime : int
   
    Returns
    -------
    Rpeaks: DataFrame
        containing all the peak values for the specified lead rhythm
    '''
    #Rpeaks, _ = signal.find_peaks(converted_signal, height= 0.6 * max(converted_signal), distance = round(SampleFrequencyTime *0.200)) #500*0.400 is about 200 which is found based on trial
    ma_qrs_lead, _ = signal.find_peaks(ma_ecg_lead, height= np.mean(ma_ecg_lead), distance = round(SampleFrequencyTime *0.200))
  
    return ma_qrs_lead


#%%
def QRSpeaks(converted_signal,SampleFrequencyTime):
    '''
    This function finds the QRSpeak values for a specific lead in the signal.
    Parameters
    ----------
    converted_signal : DataFrame
        (converted_signal = convert_ECG(df_ecg)[lead])
    SampleFrequencyTime : int
   
    Returns
    -------
    Rpeaks: DataFrame
        containing all the peak values for the specified lead rhythm
    '''
    Rpeaks, _ = signal.find_peaks(converted_signal, height= 0.6 * max(converted_signal), distance = round(SampleFrequencyTime *0.200)) #500*0.400 is about 200 which is found based on trial
  
    return Rpeaks


#%%
def generate_beat_labels(index_Rtops):
    '''
    This function generates beat labes based on the number of peak values taken from xml files.
    
    Parameters
    ----------
    index_Rtops : dataFrame
   
    Returns
    -------
    beat_labels: list
        list of beat labels
    '''
    
    beat_labels = [f'Beat_{i}' for i in range(len(index_Rtops))]
    return beat_labels


#%%
#from Margot's code with minor changes
def TCRT_parameter(energy_values_signal, index_t_TS, index_t_RS, indexToff, DC_norm_S_T_matrix, DC_norm_S_QRS_matrix, index_t_RS_a, index_t_RE_a, constant):
    '''
    Function to calculate the Total Cosine R to T. Defined as: mean angle between the vectors in the QRS complex inbetween t_RS_a and t_RE_a time interval, and the vector with maximum T wave energy.

    Parameters
    ----------
    energy_values_signal : list
        list with the ECG energy of the 3 most important dimensions per   point.
    index_t_TS : int
        index of the t_TS sample point.
    index_t_RS : int
        index of the t_RS sample point..
    indexToff : int
        index of the Toffset sample point.
    DC_norm_S_T_matrix : array of float64
        DC compensated, normalised signal matrix of the T wave.
    DC_norm_S_QRS_matrix : array of float64
        DC compensated, normalised signal matrix of the QRS complex.
    index_t_RS_a : int
        index of the t_RS_a sample point.
    index_t_RE_a : int
        index of the t_RE_a sample point.
    constant: 
        This constant is needed when vector_max_energy = DC_norm_S_T_matrix[:,max_energy_point] raise an error.
    Returns
    -------
    TCRT : float
        the Total Cosine R to T, mean angle between the vectors in the QRS complex inbetween t_RS_a and t_RE_a time interval, and the vector with maximum T wave energy.
    e_T_1 : array of float64
        unit vector in the direction of maximum T-wave energy.
    e_T_2 : array of float64
        unit vector perpendicular to e_T_1.

    '''
    #determine max energy of T wave
    max_T_energy = max(energy_values_signal[index_t_TS:(indexToff+constant)])
    #determine the vector S3D at this max value 
    max_energy_point = np.where(np.array([energy_values_signal[index_t_TS:]]) == max_T_energy)[1][0]
    #determine a vector by substracting (0,0,0) from coordinates
    vector_max_energy = DC_norm_S_T_matrix[:,max_energy_point]
    #calculate length of vector to be able to determine unit vector 
    vectorlength = math.sqrt(vector_max_energy[0]**2 + vector_max_energy[1]**2 + vector_max_energy[2]**2)
    #determine unit vector 
    e_T_1 = vector_max_energy/vectorlength
    
    #define an arbitrary vector
    arb_vector = np.array([1, 0, 0])
    #take the cross product of the given unit vector and the arbitrary vector to get a vector perpendicular to both
    perp_vector = np.cross(e_T_1, arb_vector)
    #normalize the perpendicular vector to get a unit vector
    e_T_2 = perp_vector / np.linalg.norm(perp_vector)
    
    # perp_3 = np.cross(e_T_1, e_T_2)
    # e_T_3 = perp_3 /np.linalg.norm(perp_3)
    
    #Sqrs between t'RS and t'RE
    SQRStimeintervalpeakR = DC_norm_S_QRS_matrix[:, (index_t_RS_a-index_t_RS):(index_t_RE_a-index_t_RS)]
    
    angle_list = []
    for n in range(0,len(SQRStimeintervalpeakR[0])):
        vector_dot = np.array([SQRStimeintervalpeakR[0][n], SQRStimeintervalpeakR[1][n], SQRStimeintervalpeakR[2][n]])
        dot_product = np.dot(e_T_1, vector_dot) #calculate the dot product of two 2D vectors
        denumerator_uni = np.linalg.norm(e_T_1)
        denumerator2_uni = np.linalg.norm(vector_dot)
        denumerator_total = denumerator_uni * denumerator2_uni
        angleradians = math.cos(np.arccos((dot_product/denumerator_total)))
        angle_list.append(angleradians)
    
    TCRT = np.mean(angle_list)
    return TCRT, e_T_1,  e_T_2


#%%
def validate_Toffsets(Tonsets,Tpeaks,Toffsets):
    '''
    In this function the aim is to estimate a Toffset value for some beats whose Toffset values could not been detected by the 
    imported function.
    Parameters
    ----------
    Tonsets : list
        A list containing the Tonset value for each beat
    Tpeaks : list
        A list containing the Tpeak value for each beat
    Toffsets : list
        A list containing the Toffset value or nan for each beat

    
    Returns
    -------
    valid_Toffsets: list
        A list containing the Toffset values (There may exist some NaN values)
        
    '''
    valid_Toffsets = []
    for i in range(len(Toffsets)):
        if Toffsets[i] is np.nan:
            if Tonsets[i] is np.nan:
                valid_Toffset = Tpeaks[i] + 45
                valid_Toffsets.append(valid_Toffset)
            else :
                delta_dist = Tpeaks[i]-Tonsets[i]
                valid_Toffset = delta_dist + Tpeaks[i]
                valid_Toffsets.append(valid_Toffset)
        else:
            valid_Toffsets.append(Toffsets[i])
        
        # valid_Toffset_indicies = np.where(~np.isnan(valid_Toffsets))[0] #selects the valid valus
        # valid_Toffsets1 = [valid_Toffsets[i] for i in valid_Toffset_indicies]
    return valid_Toffsets

#%%

def beat_finder(converted_ecg,peaks_df,SampleFrequencyTime,lead,beat_num, plot=True):
    '''
    This function returns the a single beat information 

    Parameters
    ----------
    converted_ecg : DataFrame
        Input signal (ECG) in mV.
    peaks_df : DataFrame
        peak indexes for all leads in the rhythm ECG
    lead : str
        lead which is investigatied such as 'lead_I'.
    beat_num : int
        which beat is investigated
        
    Returns
    -------
    lead_peak : int
        index value at which the investigated beat makes it peak
    beat_lead_Ecg: Series
        Voltage information for the specific beat
    '''
    
    lead_ecg = converted_ecg[lead]
    lead_peak = peaks_df[lead][beat_num]
    #Continuous Wavelet Method
    _, waves_cwt = nk.ecg_delineate(lead_ecg, np.array(peaks_df[lead]), sampling_rate = SampleFrequencyTime, method='cwt',show=True,show_type='bounds_T')
    
    
    Toffsets = waves_cwt['ECG_T_Offsets']
    Tonsets = waves_cwt['ECG_T_Onsets']
    Tpeaks = waves_cwt['ECG_T_Peaks']
    
    valid_Toffsets = validate_Toffsets(Tonsets, Tpeaks, Toffsets)
    try: 
        Toffset = valid_Toffsets[beat_num]
        Tonset = Tonsets[beat_num]
        Tpeak = Tpeaks[beat_num]
    except IndexError as e:
        print(f'Error at finding Twave parameter in beat: {beat_num}: {e}')
    
    start_index = max(0,lead_peak-250)
    end_index = min(len(lead_ecg),lead_peak+250) #250 chosen from inspection
    beat_lead_ecg = lead_ecg[start_index:end_index]

    if plot:
        plt.figure()
        plt.grid(visible = True,which = 'both')
        plt.plot(lead_ecg[start_index:end_index],label = f'Beat at peak: {lead_peak} for {lead}')
        

        plt.scatter(lead_peak, lead_ecg[lead_peak], color ='red', s = 75, marker = '*')
        plt.axvline(x=lead_peak,color='r',linestyle='--',label='Peak')
        if Toffset != np.nan:
            plt.scatter(Toffset, lead_ecg[Toffset], color = 'black', s=75, marker = '*' )
        if Tonset != np.nan:
            plt.scatter(Tonset, lead_ecg[Tonset], color = 'darkgreen', s=75, marker = '*' )
        if Toffset != np.nan:
            plt.scatter(Tpeak, lead_ecg[Tpeak], color = 'orange', s=75, marker = '*' )
    
        plt.xlabel('time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title(f'ECG Beat for {lead}')
        plt.legend(loc='best',fontsize='x-small')
        plt.show()
     
    return Toffset,lead_peak,beat_lead_ecg,start_index,end_index

    
#%%
def SegmentBeats(converted_ecg,index_Rtops,SampleFrequencyTime,beat_num, plot=True):
    '''
    This function takes the 'lead_I' index_Rtops and segment the beats from the their RR intervals

    Parameters
    ----------
    converted_ecg : DataFrame
        Input signal (ECG) in mV.
    index_Rtops : list
        peak indexes for lead_I
    beat_num : int
        which beat is investigated
        
    Returns
    -------
    lead_peak : int
        index value at which the investigated beat makes it peak
    beat_lead_Ecg: Series
        Voltage information for the specific beat
    '''
    
    lead_ecg = converted_ecg['lead_I']
    lead_peak = index_Rtops[beat_num]

    
    #RR/2 interval does not produce a reliable endpoint for the beat.
    # boundary_end = [round((index_Rtops[i + 1]-index_Rtops[i])/3 * 2 ) for i in range(len(index_Rtops) - 1)] #2*RR/3 interval
    # boundary_start = [round((index_Rtops[i + 1]-index_Rtops[i])/2 ) for i in range(len(index_Rtops) - 1)]
    
    
    boundary_end = [round((index_Rtops[i + 1]-index_Rtops[i])/3 * 2 ) for i in range(0, len(index_Rtops) - 1)] #2*RR/3 interval
    boundary_start = [round((index_Rtops[i]-index_Rtops[i - 1])/2 ) for i in range(1, len(index_Rtops))]
    
    boundary_end_mean = round(np.array(boundary_end).mean())
    boundary_start_mean = round(np.array(boundary_start).mean())
    
    if beat_num == 0:
        boundary_starting = boundary_start_mean
        boundary_ending = boundary_end[beat_num]
    elif beat_num == len(index_Rtops) - 1:
        boundary_starting = boundary_start[beat_num-1]
        boundary_ending = boundary_end_mean
    else:
        boundary_starting = boundary_start[beat_num-1]
        boundary_ending = boundary_end[beat_num]
        
    start_index = max(0,lead_peak-boundary_starting)
    end_index = min(len(lead_ecg),lead_peak+boundary_ending) 
    beat_lead_ecg = lead_ecg[start_index:end_index]

    SampleSize=len(converted_ecg)
    samples =list(range(SampleSize))
    t = np.array([sample / SampleFrequencyTime*1000 for sample in samples])   # transform to ms units
    t_beat = t[start_index:end_index]

    if plot:
        plt.figure()
        plt.grid(visible = True,which = 'both')
        plt.plot(t_beat, beat_lead_ecg,label = f'Beat_{beat_num} at peak: {lead_peak} for lead_I')
        

        #plt.scatter(lead_peak, lead_ecg[lead_peak], color ='red', s = 75, marker = '*')
        plt.axvline(x=t[lead_peak],color='r',linestyle='--',label='Peak')
    
        plt.xlabel('time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title('ECG Beat for lead_I')
        plt.legend(loc='best',fontsize='x-small')
        plt.show()
        
    return start_index, end_index

#%% 
#this corde is from Margot's code:
    
def plot2d_and_3d(VCG, filename, duration, SavePath, save=1):
    '''
    Creates a figure and plots the VCG signal obtained by Kors matrix multiplication in both 2D and 3D.

    Parameters
    ----------
    VCG : DataFrame
        DataFrame containing the X, Y, and Z Frank leads.
    filename : str
        The name of the file being processed.
    duration : str
        The duration of the data: should be equal to rhythm or median.
    SavePath : str
        The path where the outputted image is to be saved.
    save : int, optional
        Indicates to save the plot or not.
        1 - Save the plot (default).
        0 - Do not save the plot.
        
    Returns
    ----------
    None
       This function does not return any value. It creates a plot of the 2D and 3D planes instead. 
        
    '''
    fig = plt.figure(figsize=plt.figaspect(2.))
    manager = plt.get_current_fig_manager()
    #manager.full_screen_toggle() 
    fig.suptitle(filename[0:-4]+ 'Kors '+ duration)
    # First subplot
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(VCG[0], VCG[1])
    ax.grid(True)
    ax.set_title('Kors Frontal Plane')
    ax.set_xlabel('X lead: Amplitude (mV)')
    ax.set_ylabel('Y lead: Amplitude (mV)')
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(VCG[0], VCG[2])
    ax.grid(True)
    ax.set_xlabel('X lead: Amplitude (mV)')
    ax.set_ylabel('Z lead: Amplitude (mV)')
    ax.set_title('Kors Tranverse Plane')
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(VCG[1],VCG[2])
    ax.grid(True)
    ax.set_title('Kors Sagittal Plane')
    ax.set_xlabel('Y lead: Amplitude (mV)')
    ax.set_ylabel('Z lead: Amplitude (mV)')
    # Second subplot
    ax = fig.add_subplot(2, 3, (4, 6), projection='3d')
    surf = ax.plot3D(VCG[0],VCG[1],VCG[2])
    ax.set_title('VCG_Kors')
    ax.set_xlabel('X lead: Amplitude (mV)')
    ax.set_ylabel('Y lead: Amplitude (mV)')
    ax.set_zlabel('Z lead: Amplitude (mV)')
    if save==1:
        plt.savefig(SavePath+'\\'+filename[0:-4] + duration + 'VCG'+ '.png')
    plt.show()
    
#%%
#From Margot's code:
    
def distcalc(originwanted, inDict, VCG, indexQonset, indexQoffset, indexToffset):
    '''
    Function to calculate the vectors for QRS complex and T wave at maximum amplitude. Needed to be able to obtain the Spatial QRS-T angle. 

    Parameters
    ----------
    originwanted : array of float64
        Array containing the three floats indicating X, Y, and Z coordinate points for the determined origin.
    inDict : dict
        Dictionary containing data from input xml files.
    VCG : DataFrame
        DataFrame containing the X, Y, and Z Frank leads.

    Returns
    -------
    rpeakamplitude : array of float64
        coordinates of the moment of maximum QRS amplitude compared to origin calculated.
    tpeakamplitude : array of float64
        coordinates of the moment of maximum T amplitude compared to origin calculated.
    indexQonset : int
        index Q onset.
    indexQoffset : int
        index Q offset.
    indexToffset : int
        index T offset.
    indexpeakQRS : int
        index of calculated peak QRS time point.
    indexpeakT : int
        index of calculated peak T time point.

    '''
    originx = originwanted[0]
    originy = originwanted[1]
    originz = originwanted[2]
    
    distQRS = {}
    distT = {}
    for i in range(indexQonset, indexQoffset + 1, 1):
        distQRS[i] = math.sqrt((VCG[0].iloc[i] - originx)**2 + (VCG[1].iloc[i] - originy)**2 + (VCG[2].iloc[i] - originz)**2)
    valuepeakQRS = max(distQRS.values())
    listQRSvalues = list(distQRS.values())
    indexpeakQRS = listQRSvalues.index(valuepeakQRS) + indexQonset #+indexQonset since indexes are not kept
    for l in range(indexQoffset, indexToffset + 1, 1):
        distT[l] = math.sqrt((VCG[0].iloc[l] - originx)**2 + (VCG[1].iloc[l] - originy)**2 + (VCG[2].iloc[l] - originz)**2)
    valuepeakT = max(distT.values())
    listTvalues = list(distT.values())
    indexpeakT = listTvalues.index(valuepeakT) + indexQoffset #+indexTonset since indexes are not kept
    rpeakamplitude = np.array(VCG.loc[indexpeakQRS]) #Get the beloging values of the VCG signal at the index of the maximal distance in 3D to origin
    tpeakamplitude = np.array(VCG.loc[indexpeakT])
    return rpeakamplitude, tpeakamplitude, indexQonset, indexQoffset, indexToffset, indexpeakQRS, indexpeakT

#%%
#from Margot's code
def plotprojection_QRSloop(filename, duration, vector1, vector2, originwanted, VCG, beat_num, radius=0.2, fig=None, colour='C0'):
    '''
    Plot arc between two given vectors in 3D space.
    Calculate vector between two vector end points, and the resulting spherical angles for various points along 
    this vector. From this, derive points that lie along the arc between vector1 and vector2
        
    Parameters
    ----------
    vector1 : list
        Coordiantes of first vector in X, Y, and Z.
    vector2 : list
        Coordiantes of second vector in X, Y, and Z.
    originwanted : array of float64
        Array containing the three floats indicating X, Y, and Z coordinate points for the determined origin.
    VCG : DataFrame
        DataFrame containing the X, Y, and Z Frank leads.
    radius : TYPE, optional
        radius. The default is 0.2.
    fig : str, optional
        fig is None or not. The default is None.
    colour : str, optional
        Colour to plot. The default is 'C0'.

    Returns
    -------
    fig : figure, plot
        figure containing the different inputed vectors and the angle.
    vectorneededazimuth : list
        list containing coordiantes of the needed azimuth vector.

    '''
    #first subplot:
    fig = plt.figure(figsize=plt.figaspect(2.))
    manager = plt.get_current_fig_manager()
    #manager.full_screen_toggle() 
    fig.suptitle(filename[0:-4]+ 'Kors '+ duration)
    # First subplot
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(VCG[0], VCG[1])
    ax.grid(True)
    ax.set_title('Kors Frontal Plane')
    ax.set_xlabel('X lead: Amplitude (mV)')
    ax.set_ylabel('Y lead: Amplitude (mV)')
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(VCG[0], VCG[2])
    ax.grid(True)
    ax.set_xlabel('X lead: Amplitude (mV)')
    ax.set_ylabel('Z lead: Amplitude (mV)')
    ax.set_title('Kors Tranverse Plane')
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(VCG[1],VCG[2])
    ax.grid(True)
    ax.set_title('Kors Sagittal Plane')
    ax.set_xlabel('Y lead: Amplitude (mV)')
    ax.set_ylabel('Z lead: Amplitude (mV)')
    
    #second subplot
    
    vector = [i-j for i, j in zip(vector1, vector2)]
    vector_points_direct = [(vector2[0]+vector[0]*l, vector2[1]+vector[1]*l, vector2[2]+vector[2]*l) for l in np.linspace(0, 1)]
    vector_phis = [math.atan2(vector_point[1], vector_point[0]) for vector_point in vector_points_direct]
    vector_thetas = [math.acos(vector_point[2]/np.linalg.norm(vector_point)) for vector_point in vector_points_direct]

    vector_points_arc = [(radius*sin(theta)*cos(phi), radius*sin(theta)*sin(phi), radius*cos(theta))
                    for theta, phi in zip(vector_thetas, vector_phis)]
    vector_points_arc.append((0, 0, 0))

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        #ax = fig.gca()
        ax = fig.add_subplot(2, 3, (4, 6), projection='3d')
        
    #from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    #points_collection = Poly3DCollection([vector_points_arc], alpha=0.4)
    #points_collection.set_facecolor(colour)
    surf = ax.plot3D(VCG[0],VCG[1],VCG[2])
    #ax.add_collection3d(points_collection)
    
    ax.scatter(originwanted[0], originwanted[1], originwanted[2], color="red")
    ax.quiver(originwanted[0], originwanted[1], originwanted[2], vector1[0], vector1[1], vector1[2], color="red", label ='QRS peak')
    ax.quiver(originwanted[0], originwanted[1], originwanted[2], vector2[0], vector2[1], vector2[2], color="cyan", label = 'T peak')
    #this part of the code plots the orthogonal projected vector
    ax.legend(loc='best')
    #ax.quiver(originwanted[0], originwanted[1], originwanted[2], vector1[0], originwanted[1], vector1[2], color="pink")
    #This part of the code plots the normal vector of the orthogonal projected vector
    xvalue = [originwanted[0], 1.3]
    yvalue = [originwanted[1], originwanted[1]]
    zvalue = [originwanted[2], originwanted[2]]
    vectorneededazimuth = np.array([xvalue[1], yvalue[1], zvalue[1]]) #the orange vector
    #ax.plot(xvalue, yvalue, zvalue)
    ax.set_title(f'VCG_Kors for beat: {beat_num}')
    ax.set_xlabel('X lead: Amplitude (mV)')
    ax.set_ylabel('Y lead: Amplitude (mV)')
    ax.set_zlabel('Z lead: Amplitude (mV)')
    return fig, vectorneededazimuth

#%%
        
# plt.close('all')
# folderpath = 'C:\\Users\\defne.odabasi\\Documents\\ECG\\ECG - Anonieme Data'
# filename = 'C14_ECG_19.xml'

# ind_=filename.find('_')         #text for _ is subfolder
# subfolder=filename[0:ind_]
# filepath=folderpath+'\\'+subfolder+'\\'+filename
# with open(filepath, "rb") as f:
#     inDict = xmltodict.parse(f)
#     f.close()
        
# #rhythm
# sub_dict=inDict['RestingECG']['Waveform'][1]['LeadData']
# SampleFrequencyTime = int(inDict['RestingECG']['Waveform'][0]['SampleBase']) 
# ecg_data={}
# for ld in sub_dict: 
#     ecg_data.update(decode_lead(ld)) 
    
# df_ecg = pd.DataFrame(ecg_data)

# lead = 'lead_I' #specify the lead
# converted_signal = functions_beat_to_beat.convert_ECG(df_ecg)[lead]
# ma_ecg = functions_beat_to_beat.pan_tompkins_qrs_detection(converted_signal,SampleFrequencyTime)

# ma_qrs = functions_beat_to_beat.QRSpeaks(ma_ecg, SampleFrequencyTime)
# Rpeaks = functions_beat_to_beat.QRSpeaks(converted_signal,SampleFrequencyTime)

# #Continuous Wavelet Method
# _, waves_cwt = nk.ecg_delineate(converted_signal, Rpeaks, sampling_rate = SampleFrequencyTime, method='cwt',show=True,show_type='bounds_T')

# #For NaN values and estimation might be useful
# Toffsets = waves_cwt['ECG_T_Offsets']
# Tonsets = waves_cwt['ECG_T_Onsets']
# Tpeaks = waves_cwt['ECG_T_Peaks']

# #validate_Toffsets(Tonsets, Tpeaks, Toffsets)
# # if still there are nan values in the valid_Toffsets
# valid_Toffsets = functions_beat_to_beat.validate_Toffsets(Tonsets, Tpeaks, Toffsets)


# valid_Toffset_indicies = np.where(~np.isnan(valid_Toffsets))[0] #selects the valid valus
# valid_Toffsets1 = [valid_Toffsets[i] for i in valid_Toffset_indicies]

# #plot for rhythm ecg
# plt.figure(figsize= (20,4), dpi = 100)
# plt.xticks(np.arange(0,len(converted_signal)+1,150))
# plt.plot(converted_signal, color = 'blue')
# plt.scatter(Rpeaks, converted_signal[Rpeaks], color ='red', s = 50, marker = '*')
# plt.scatter(valid_Toffsets1, converted_signal[valid_Toffsets1], color = 'green', s=50, marker = '*' ) 
# #plt.scatter(Tonsets, converted_signal[Tonsets], color = 'yellow', s=50, marker = '*' ) 
# #plt.scatter(Tpeaks, converted_signal[Tpeaks], color = 'purple', s=50, marker = '*' ) 
# plt.xlabel('ms')
# plt.ylabel('MLIImV')
# plt.title('R peak Locations')

#%%
