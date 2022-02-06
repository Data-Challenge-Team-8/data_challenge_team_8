import streamlit as st
import hydralit_components as hc
from web.categories.landing_page import LandingPage
from web.categories.data_loader import DataLoader
from web.categories.exploratory_data_analysis import ExploratoryDataAnalysis
from web.categories.sepsis_research import SepsisResearch
from web.categories.ts_analysis import TimeSeriesAnalysis


def create_app():
    #############
    st.set_page_config(layout='wide', initial_sidebar_state='expanded', menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     })

    # specify the primary menu definition
    menu_data = [
        {'icon': "far fa-copy",'label': "Data Loader"}, #<i class="far fa-dove" style="color: #339af0;"></i>
        {'icon': "far fa-chart-bar", 'label': "Exploratory Data Analysis"},
        {'icon': "far fa-address-book", 'label': "Sepsis Research"},
        {'icon': "fas fa-tachometer-alt", 'label': "Time Series Analysis"}
    ]
    # we can override any part of the primary colors of the menu
    over_theme = {'txc_inactive': 'white',
                  'menu_background': 'grey',
                  'txc_active': 'black',
                  'option_active': 'white'}

    menu_id = hc.nav_bar(menu_definition=menu_data,
                         override_theme=over_theme,
                         home_name='Home',
                         hide_streamlit_markers=False,
                         sticky_nav=True,  # at the top or not
                         sticky_mode='not-jumpy')


    if menu_id == 'Home':
        landing_page = LandingPage()
    if menu_id == 'Data Loader':
        data_loader = DataLoader()
    if menu_id == 'Exploratory Data Analysis':
        expl_ana = ExploratoryDataAnalysis()
    if menu_id == 'Sepsis Research':
       math_stat = SepsisResearch()
    if menu_id == 'Time Series Analysis':
        math_stat = TimeSeriesAnalysis()

create_app()
