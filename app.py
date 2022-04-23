import streamlit as st
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K 
tf.compat.v1.disable_eager_execution()
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

st.set_page_config(page_title="DMSP", layout="centered")
features=['sc_aacgm_lat', 'id_sc', 'sc_aacgm_ltime',
       'sin_sc_aacgm_ltime', 'cos_sc_aacgm_ltime', 'f107_30min', 
       
       'f107_45min','f107_1hr', 'f107_10min', 'f107', 'f107_6hr', 'f107_3hr', 
       'ae_1hr','au', 'ae_30min', 'ae_45min', 'al_1hr', 'ae', 
       'au_1hr', 'pc_1hr','au_45min', 'al_45min', 'ae_10min', 'al_3hr',
       'al_10min', 'sin_doy','pc', 'pc_30min', 'au_30min', 'pc_10min',
       'al_30min', 'pc_45min','newell_3hr', 'au_10min', 'ae_6hr', 'cos_doy',
       'au_3hr', 'bz_1hr','newell_1hr', 'ae_3hr', 'borovsky_1hr', 'newell_45min',
       'vx_6hr','vsw_30min', 'borovsky_3hr', 'bz_30min', 'vx', 'al',
       'vsw_10min','newell_6hr', 'bz_6hr', 'vsw_45min', 'vx_10min', 'pc_3hr',
       'psw_3hr','bz_45min', 'vsw', 'au_6hr', 'symh_1hr', 'symh_10min', 
       'vsw_1hr','vsw_6hr', 'vx_30min', 'bx_6hr', 'vx_1hr', 'by_1hr',
       
       'borovsky_10min']
st.markdown("<h1 style='text-align: center;'>DMSP-Particle-Precipitate-Flux-Prediction-Mesoscale </h1>", unsafe_allow_html=True)     
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")  

        sc_aacgm_lat=st.number_input('sc_aacgm_lat')
        id_sc=st.number_input('id_sc')
        sc_aacgm_ltime=st.number_input('sc_aacgm_ltime')
        sin_sc_aacgm_ltime=st.number_input('sin_sc_aacgm_ltime')
        cos_sc_aacgm_ltime=st.number_input('cos_sc_aacgm_ltime')
        f107_30min=st.number_input('f107_30min')


        f107_45min=st.number_input('f107_45min')
        f107_1hr=st.number_input('f107_1hr')
        f107_10min=st.number_input('f107_10min')
        f107=st.number_input('f107')
        f107_6hr=st.number_input('f107_6hr')
        f107_3hr=st.number_input('f107_3hr')

  
        ae_1hr=st.number_input('ae_1hr')    
        au=st.number_input('au')   
        ae_30min=st.number_input('ae_30min')   
        ae_45min=st.number_input( 'ae_45min')   
        al_1hr=st.number_input('al_1hr')   
        ae=st.number_input('ae')   

        au_1hr=st.number_input('au_1hr')   
        pc_1hr=st.number_input('pc_1hr')  
        au_45min=st.number_input('au_45min')  
        al_45min=st.number_input('al_45min')  
        ae_10min=st.number_input('ae_10min')  
        al_3hr=st.number_input('al_3hr')  


        al_10min=st.number_input('al_10min') 
        sin_doy=st.number_input('sin_doy') 
        pc=st.number_input('pc') 
        pc_30min=st.number_input('pc_30min') 
        au_30min=st.number_input('au_30min') 
        pc_10min=st.number_input( 'pc_10min') 

     
        al_30min=st.number_input('al_30min') 
        pc_45min=st.number_input( 'pc_45min') 
        newell_3hr=st.number_input( 'newell_3hr') 
        au_10min=st.number_input( 'au_10min') 
        ae_6hr=st.number_input(  'ae_6hr') 
        cos_doy=st.number_input('cos_doy') 
  
        au_3hr=st.number_input('au_3hr') 
        bz_1hr=st.number_input('bz_1hr') 
        newell_1hr=st.number_input('newell_1hr') 
        ae_3hr=st.number_input('ae_3hr') 
        borovsky_1hr=st.number_input( 'borovsky_1hr') 
        newell_45min=st.number_input('newell_45min')  


 
        vx_6hr=st.number_input('vx_6hr') 
        vsw_30min=st.number_input( 'vsw_30min') 
        borovsky_3hr=st.number_input('borovsky_3hr') 
        bz_30min=st.number_input('bz_30min') 
        vx=st.number_input('vx') 
        al=st.number_input('al') 

       
        vsw_10min=st.number_input('vsw_10min') 
        newell_6hr=st.number_input('newell_6hr') 
        bz_6hr=st.number_input( 'bz_6hr') 
        vsw_45min=st.number_input('vsw_45min') 
        vx_10min=st.number_input('vx_10min') 
        pc_3hr=st.number_input( 'pc_3hr') 


       
        psw_3hr=st.number_input('psw_3hr') 
        bz_45min=st.number_input( 'bz_45min') 
        vsw=st.number_input( 'vsw') 
        au_6hr=st.number_input( 'au_6hr') 
        symh_1hr=st.number_input( 'symh_1hr') 
        symh_10min=st.number_input('symh_10min') 

      
        vsw_1hr=st.number_input('vsw_1hr') 
        vsw_6hr=st.number_input( 'vsw_6hr') 
        vx_30min=st.number_input( 'vx_30min') 
        bx_6hr=st.number_input( 'bx_6hr') 
        vx_1hr=st.number_input( 'vx_1hr',) 
        by_1hr=st.number_input('by_1hr') 
        borovsky_10min=st.number_input('borovsky_10min') 

        submit = st.form_submit_button("Predict")
        if submit:
            model=load_model('Model/keras_bestmodel.h5')  
            data=np.array(features).reshape(1,-1)
            pred=model.predict(data)
            result = 10**pred
            st.write(f"The predicted severity is: {result}")    

if __name__ == '__main__':
    main()

         