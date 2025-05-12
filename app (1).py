import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import calplot
import holidays
from datetime import datetime
import plotly.express as px
import calendar
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.patches import Rectangle
import geopandas as gpd
from shapely.geometry import LineString
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import requests
from pyproj import Transformer
import warnings
import pickle
import io
import gdown


st.set_page_config(layout="wide", page_title="Traffic", page_icon="🚗")


@st.cache_data
def load_final_df():
    url = "https://drive.google.com/uc?id=1Ju6ctXTKOIAeUsgXX8J_GCE680kW-Cvg"
    output = "final_df.parquet"
    gdown.download(url, output, quiet=False)
    return pd.read_parquet(output)

@st.cache_resource
def load_avg_speed_model():
    url = "https://drive.google.com/uc?export=download&id=1AqCCQxXkgk6X6_YEtBLkfaxOTxyfLDiP"
    response = requests.get(url)
    response.raise_for_status()
    return pickle.load(io.BytesIO(response.content))

@st.cache_resource
def load_volume_model():
    url = "https://drive.google.com/uc?export=download&id=1_nqP2bt3Qqh63Cl7B9h-gMpaV4gJLIZr"
    response = requests.get(url)
    response.raise_for_status()
    return pickle.load(io.BytesIO(response.content))


final_df = load_final_df()
speed_model = load_avg_speed_model()
volume_model = load_volume_model()

with st.spinner("Loading dashboard..."):
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "Homepage",
        "Overview",
        'Traffic Analysis by Country and Direction',
        "Traffic Trend Analysis",
        "Vehicle Type Contribution to Congestion",
        "Holiday Traffic Trends",
        "Traffic Calendar",
        "Speed Behavior Analysis",
        'Map For Live Traffic Information',
        'Traffic Prediction',
        'Volume Prediction',
        'Limitations'
        ])

    with tab1:
        st.title("Traffic Analysis")
        st.markdown("The datasets used for this project were collected from Lithuanian Open Data Portal and Traffic Informational Center.")
        st.divider()
        st.markdown("Our dataset contains traffic records measured by sensors installed along various Lithuanian highways and roads. The period covered by the data spans from 2018 to the end of 2023. ")

        st.subheader("Random Sample (20 Rows)")
        st.dataframe(final_df.sample(20))
        st.divider()
        st.subheader("Summary Statistics")
        st.dataframe(final_df.describe())

    with tab2:
        vehicles_per_day = final_df.groupby('Vehicle_Date')['Number_of_Vehicles'].mean()
        max_vehicles_day = vehicles_per_day.idxmax()
        min_vehicles_day = vehicles_per_day.idxmin()

        vehicles_per_month = final_df.groupby('Month')['Number_of_Vehicles'].mean()
        max_vehicles_month = vehicles_per_month.idxmax()
        min_vehicles_month = vehicles_per_month.idxmin()

        st.subheader("Distribution of Average Speed")
        fig = px.histogram(
            final_df,
            x='Average_Speed',
            nbins=50,
            labels={'Average_Speed': 'Average Speed (km/h)'},
            opacity=0.75
        )
        fig.update_layout(xaxis_title='Average Speed (km/h)', yaxis_title='Count')
        st.plotly_chart(fig)

        st.divider()

        st.subheader("Average Number of Vehicles per Day")
        vehicles_per_day_df = vehicles_per_day.reset_index()
        fig3 = px.line(
            vehicles_per_day_df,
            x='Vehicle_Date',
            y='Number_of_Vehicles',
            labels={'Vehicle_Date': 'Date', 'Number_of_Vehicles': 'Vehicles'},
            template='plotly_white'
        )
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("See more"):
             st.markdown(f"""
                 - **Day with Max Vehicles: 2018-07-08 (Sunday)**
                 - **Day with Min Vehicles: 2020-04-12 (Sunday)**
                 - **Month with Max Avg Vehicles: 2018-September**
                 - **Month with Min Avg Vehicles: 2018-February**
             """)

        st.divider()

        st.subheader("Average Number of Vehicles per Month (Across All Years)")
        avg_veh_month = final_df.groupby('Month')['Number_of_Vehicles'].mean()

        fig_bar, ax = plt.subplots(figsize=(12, 6))
        avg_veh_month.plot(kind='bar', ax=ax)
        ax.set_ylabel('Average Number of Vehicles')
        ax.set_xlabel('Month')
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
        st.pyplot(fig_bar)

        st.divider()

        pd.set_option('mode.chained_assignment', None)
        warnings.simplefilter(action='ignore', category=FutureWarning)


        def classify_road_type(road_number):
            if pd.isnull(road_number):
                return "Unknown"
            road_str = str(road_number)
            if road_str.startswith("A"):
                return "A road"
            elif road_str.isdigit():
                number = int(road_str)
                if number <= 999:
                    return "Regional road"
                elif number >= 1000:
                    return "Municipal road"
            return "Unknown"

        final_df['Road_Type'] = final_df['Road_Number'].apply(classify_road_type)

        road_stats = final_df.groupby(['Road_Type', 'Road_Number']).agg({
                'Average_Speed': 'mean',
                'Number_of_Vehicles': 'mean'
            }).reset_index()

        sorted_roads = road_stats.sort_values(by='Average_Speed')
        sorted_vehicles = road_stats.sort_values(by='Number_of_Vehicles')

        # top_slow = sorted_roads.head(5)
        # top_fast = sorted_roads.tail(5).sort_values(by='Average_Speed', ascending=False)
        # median_idx = len(sorted_roads) // 2
        # median_roads = sorted_roads.iloc[median_idx - 2: median_idx + 3]

        # least_busy = sorted_vehicles.head(5)
        # busiest = sorted_vehicles.tail(5).sort_values(by='Number_of_Vehicles', ascending=False)
        # median_busy = sorted_vehicles.iloc[median_idx - 2: median_idx + 3]

        # for df in [top_fast, top_slow, median_roads, least_busy, busiest, median_busy]:
        #     df['Label'] = df['Road_Type'] + ' ' + df['Road_Number'].astype(str)

        # st.subheader("Roads by Speed and Volume")

        # fig, axs = plt.subplots(2, 2, figsize=(20, 12))
        # slow_palette = sns.light_palette("red", n_colors=5, reverse=True)
        # median_palette_speed = sns.light_palette("green", n_colors=5, reverse=True)
        # busiest_palette = sns.light_palette("blue", n_colors=5, reverse=True)
        # least_palette = sns.light_palette("red", n_colors=5, reverse=True)
        # median_palette_traffic = sns.light_palette("green", n_colors=5, reverse=True)

        # sns.barplot(x='Average_Speed', y='Label', data=top_fast, ax=axs[0, 0], palette='Blues_r')
        # axs[0, 0].set_title('Top 5 Fastest Roads')
        # sns.barplot(x='Average_Speed', y='Label', data=top_slow, ax=axs[0, 1], palette=slow_palette)
        # axs[0, 1].set_title('Top 5 Slowest Roads')
        # sns.barplot(x='Average_Speed', y='Label', data=median_roads, ax=axs[0, 2], palette=median_palette_speed)
        # axs[1, 0].set_title('Median Roads (Middle 5)')
        # sns.barplot(x='Number_of_Vehicles', y='Label', data=busiest, ax=axs[1, 0], palette=busiest_palette)
        # axs[1, 0].set_title('Top 5 Busiest Roads')
        # sns.barplot(x='Number_of_Vehicles', y='Label', data=least_busy, ax=axs[1, 1], palette=least_palette)
        # axs[1, 1].set_title('Top 5 Least Busy Roads')
        # sns.barplot(x='Number_of_Vehicles', y='Label', data=median_busy, ax=axs[1, 2], palette=median_palette_traffic)
        # axs[1, 2].set_title('Median Busy Roads (Middle 5)')
        # plt.tight_layout()
        # st.pyplot(fig)

        st.divider()

        st.subheader("Speed vs. Traffic Volume by Road Type")

        plt.figure(figsize=(10,6))
        sns.scatterplot(data=road_stats, x='Average_Speed', y='Number_of_Vehicles', hue='Road_Type')
        #plt.title('Speed vs. Traffic Volume')
        plt.tight_layout()
        st.pyplot(plt.gcf())

        with st.expander("See more"):
             st.markdown(f"""
                 - **The peak traffic on A19 occurred on: 2021-06-04**
                 - **Peak traffic occurred near Longitude: 574434.7899999999, Latitude: 6055780.185**
             """)

        st.divider()

        st.subheader("Peak Traffic Location on A19")

        lks92_longitude = 574434.79
        lks92_latitude = 6055780.19

        transformer = Transformer.from_crs("epsg:3346", "epsg:4326", always_xy=True)
        longitude, latitude = transformer.transform(lks92_longitude, lks92_latitude)

        peak_traffic_data = pd.DataFrame({
            'Road_Number': ['A19'],
            'Average Number of Vehicles': [1627.999581],
            'Average Speed (km/h)': [82.341626],
            'Latitude': [latitude],
            'Longitude': [longitude]
        })

        fig = px.scatter_mapbox(
            peak_traffic_data,
            lat="Latitude",
            lon="Longitude",
            size="Average Number of Vehicles",
            color="Average Speed (km/h)",
            hover_name="Road_Number",
            hover_data={
                "Average Number of Vehicles": True,
                "Average Speed (km/h)": True,
                "Latitude": False,
                "Longitude": False
            },
            color_continuous_scale="Viridis",
            size_max=20,
            zoom=10
        )

        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig)

        st.divider()

        st.subheader("Traffic Breakdown by Road Type")

        a_roads = road_stats[road_stats['Road_Type'] == 'A road']
        regional_roads = road_stats[road_stats['Road_Type'] == 'Regional road']
        municipal_roads = road_stats[road_stats['Road_Type'] == 'Municipal road']

        fig, axs = plt.subplots(3, 2, figsize=(18, 18))

        def plot_bar(ax, df, y_col, title, ylabel, color):
                x_pos = np.arange(len(df))
                ax.bar(x_pos, df[y_col], color=color)
                ax.set_title(title)
                ax.set_xlabel('Road Number')
                ax.set_ylabel(ylabel)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(df['Road_Number'], rotation=60)

        plot_bar(axs[0, 0], a_roads, 'Average_Speed', 'Average Speed on A Roads', 'Average Speed', 'lightblue')
        plot_bar(axs[0, 1], a_roads, 'Number_of_Vehicles', 'Vehicle Volume on A Roads', 'Number of Vehicles', 'lightgreen')
        plot_bar(axs[1, 0], regional_roads[::2], 'Average_Speed', 'Average Speed on Regional Roads', 'Average Speed', 'lightblue')
        plot_bar(axs[1, 1], regional_roads[::2], 'Number_of_Vehicles', 'Vehicle Volume on Regional Roads', 'Number of Vehicles', 'lightgreen')
        plot_bar(axs[2, 0], municipal_roads[::3], 'Average_Speed', 'Average Speed on Municipal Roads', 'Average Speed', 'lightblue')
        plot_bar(axs[2, 1], municipal_roads[::3], 'Number_of_Vehicles', 'Vehicle Volume on Municipal Roads', 'Number of Vehicles', 'lightgreen')

        # plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        st.pyplot(fig)


    with tab3:
        # st.markdown("""
        #         This dashboard analyzes traffic volume and average speeds based on:
        #         - License plate country (Top 5)
        #         - Road direction (N vs. T)

        #         Data has been cleaned to remove unknown or invalid license plate countries.
        #         """)
        def clean_license_plate_country(df):
            if 'License_Plate_Country' in df.columns:
                df = df[df['License_Plate_Country'].notna()]  # remove NaNs
                df = df[~df['License_Plate_Country'].str.strip().str.lower().isin(['unknown', 'x'])]
            return df

        df = final_df
        df = clean_license_plate_country(df)

        expected_cols = ['License_Plate_Country', 'Road_Number', 'Number_of_Vehicles', 'Average_Speed']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing expected columns in dataset: {missing_cols}")
            st.stop()

        grouped = df.groupby(['License_Plate_Country', 'Road_Number']).agg(
            vehicle_count=('Number_of_Vehicles', 'sum'),
            average_speed=('Average_Speed', 'mean'),
            std_speed=('Average_Speed', 'std')
        ).reset_index()

        top_countries = df["License_Plate_Country"].value_counts().head(5).index.tolist()
        top_roads = grouped.groupby("Road_Number")["vehicle_count"].sum().nlargest(3).index.tolist()

        filtered = grouped[
            (grouped["License_Plate_Country"].isin(top_countries)) &
            (grouped["Road_Number"].isin(top_roads))
        ]
        # Average Speed Plot
        st.subheader("Average Speed by Country on Top 3 Roads")
        g1 = sns.FacetGrid(filtered, col="Road_Number", col_wrap=3, height=4, sharey=False)
        g1.map_dataframe(sns.barplot, x="License_Plate_Country", y="average_speed", palette="coolwarm", order=top_countries)
        g1.set_titles("Road: {col_name}")
        g1.set_axis_labels("Country", "Average Speed (km/h)")
        g1.fig.subplots_adjust(top=0.85)
        #g1.fig.suptitle("Average Speed by Country on Top 3 Roads", fontsize=14)
        st.pyplot(g1.fig)

        # Volume Plot
        st.subheader("Traffic Volume by Country on Top 3 Roads")
        g2 = sns.FacetGrid(filtered, col="Road_Number", col_wrap=3, height=4, sharey=False)
        g2.map_dataframe(sns.barplot, x="License_Plate_Country", y="vehicle_count", palette="viridis", order=top_countries)
        g2.set_titles("Road: {col_name}")
        g2.set_axis_labels("Country", "Vehicle Count")
        g2.fig.subplots_adjust(top=0.85)
        #g2.fig.suptitle("Traffic Volume by Country on Top 3 Roads", fontsize=14)
        st.pyplot(g2.fig)
        # st.markdown("""
        # The two charts together provide a comparative analysis of traffic behavior by the top 5 most common countries in the dataset (LT, PL, LV, UA, EE) across the three most frequently used roads (A1, A5, A8). The first chart shows average speeds, while the second chart displays traffic volume. One key insight is that Lithuania (LT) dominates traffic volume on all three roads, especially on A1, where it vastly outnumbers vehicles from other countries. However, drivers from countries like Ukraine (UA) and Estonia (EE) tend to drive at higher average speeds than Lithuanians on each road.
        # """)

        st.divider()
        # --- GAWA'S SECTION 2: DIRECTIONAL ANALYSIS ---

        #st.subheader("Traffic by Direction Across Roads")
        # st.markdown("""
        # **Based on cleaned data that removes unknown or invalid license plate countries, how does vehicle traffic volume vary by direction of travel (`VEH_DIRECTION`) across different roads? Which road–direction pairs have the highest vehicle counts, and what does this reveal about directional traffic patterns in Lithuania?**

        # *Note: This chart shows the true busiest roads after removing 'x' and 'unknown' entries.*
        # """)


        # Rename for consistency
        df_subset = df[["Vehicle_Direction", "Average_Speed", "Road_Number"]].dropna()
        df_subset = df_subset.rename(columns={
            "Vehicle_Direction": "VEH_DIRECTION",
            "Average_Speed": "AVG_SPEED",
            "Road_Number": "KEL_NUMERIS"
        })
        combined_df = df_subset.copy()

        direction_summary = combined_df.groupby(["VEH_DIRECTION", "KEL_NUMERIS"]).agg(
            average_speed=("AVG_SPEED", "mean"),
            vehicle_count=("AVG_SPEED", "count")
        ).reset_index()

        direction_sorted = direction_summary.sort_values(by="vehicle_count", ascending=False)

        # Traffic Volume by Direction
        st.subheader("Top Roads by Vehicle Count and Direction")
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        sns.barplot(data=direction_sorted.head(15), x="KEL_NUMERIS", y="vehicle_count", hue="VEH_DIRECTION", ax=ax1)
        #ax1.set_title("Top Roads by Vehicle Count and Direction")
        ax1.set_xlabel("Road Number")
        ax1.set_ylabel("Vehicle Count")
        ax1.legend(title="Direction")
        st.pyplot(fig1)

        # Average Speed by Direction
        st.subheader("Average Speed by Direction on Top Roads")
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        sns.barplot(data=direction_sorted.head(15), x="KEL_NUMERIS", y="average_speed", hue="VEH_DIRECTION", palette="coolwarm", ax=ax2)
        #ax2.set_title("Average Speed by Road and Direction")
        ax2.set_xlabel("Road Number")
        ax2.set_ylabel("Average Speed (km/h)")
        ax2.legend(title="Direction")
        st.pyplot(fig2)
        # st.markdown("""
        # The two bar charts together offer a view of traffic patterns on Lithuania’s busiest roads by comparing both vehicle count and average speed across travel directions. From the first chart, we observe that roads like A5, A6, and A1 experience the highest traffic volumes, with noticeable imbalances between directions—for instance, A6 sees more northbound traffic than southbound. The second chart reveals how speeds also vary by direction: on A1, northbound traffic tends to move significantly faster than southbound, while the opposite trend appears on A6. Some roads like A8 show both moderate volume and a sharp contrast in speed between directions, possibly indicating congestion or infrastructure issues in one direction.
        # """)

    with tab4:
        duplicates = final_df[final_df.duplicated(subset=["Range_Identifier", "Number_of_Vehicles", "Average_Speed"], keep=False)]
        df_no_duplicates = final_df.drop_duplicates(subset=["Range_Identifier", "Number_of_Vehicles", "Average_Speed"])

        grouped_df = df_no_duplicates.groupby(["Range_Identifier", "Road_Number"], as_index=False).agg({
            "Number_of_Vehicles": "mean",
            "Average_Speed": "mean",
            "Range_Latitude_ETRS89": "first",
            "Range_Longitude_ETRS89": "first"
        })

        road_stats_avg = grouped_df.groupby("Road_Number").agg({
            "Number_of_Vehicles": "mean",
            "Average_Speed": "mean",
            "Range_Latitude_ETRS89": "first",
            "Range_Longitude_ETRS89": "first"
        }).reset_index()

        road_stats_avg.rename(columns={
            "Number_of_Vehicles": "Average Number of Vehicles",
            "Average_Speed": "Average Speed (km/h)",
            "Range_Latitude_ETRS89": "Latitude",
            "Range_Longitude_ETRS89": "Longitude"
        }, inplace=True)

        st.title("Traffic Data Visualization")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Average Traffic Data per Road")
            fig1 = px.scatter_mapbox(
                road_stats_avg,
                lat="Latitude",
                lon="Longitude",
                size="Average Number of Vehicles",
                color="Average Speed (km/h)",
                hover_name="Road_Number",
                hover_data={
                    "Average Number of Vehicles": True,
                    "Average Speed (km/h)": True,
                    "Latitude": False,
                    "Longitude": False
                },
                color_continuous_scale="Viridis",
                size_max=20,
                zoom=5.5
            )
            fig1.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig1)

        with col2:
            st.subheader("Clustered Road Segments by Traffic Profile")
            clustering_df = road_stats_avg[[
                "Average Number of Vehicles",
                "Average Speed (km/h)"
            ]].copy()
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(clustering_df)
            kmeans = KMeans(n_clusters=4, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            road_stats_avg["Cluster"] = clusters

            fig2 = px.scatter_mapbox(
            road_stats_avg,
            lat="Latitude",
            lon="Longitude",
            color="Cluster",
            size="Average Number of Vehicles",
            hover_name="Road_Number",
            hover_data={
                "Average Number of Vehicles": True,
                "Average Speed (km/h)": True,
                "Cluster": True,
                "Latitude": False,
                "Longitude": False
            },
            zoom=5.5,
            mapbox_style="open-street-map"
        )
            st.plotly_chart(fig2)

    with tab5:
        st.subheader("🚚 Vehicle Type Contribution to Congestion")
        def vehicle_type_congestion(df):
            df = df[df['Vehicle_Type'].str.lower() != 'unknown']
            daily = df.groupby(['Vehicle_Date', 'Vehicle_Type', 'Road_Number'])['Number_of_Vehicles'].sum().reset_index()
            agg = daily.groupby(['Vehicle_Type', 'Road_Number'])['Number_of_Vehicles'].mean().reset_index()

            road_options = sorted(agg['Road_Number'].astype(str).unique())
            selected_road = st.selectbox("Select a Road Number:", road_options, key='road')

            filtered = agg[agg['Road_Number'].astype(str) == selected_road]

            fig = px.bar(
                filtered.sort_values('Number_of_Vehicles', ascending=False),
                x='Vehicle_Type',
                y='Number_of_Vehicles',
                title=f'Average Vehicles per Day by Type on Road {selected_road}'
            )

            st.plotly_chart(fig, use_container_width=True)

        vehicle_type_congestion(final_df)

    with tab6:
        final_df['Holiday'] = final_df['Vehicle_Date'].isin([
        '2018-12-24', '2019-12-24', '2020-12-24','2021-12-24','2022-12-24','2023-12-24'
        '2018-12-25', '2019-12-25', '2020-12-25','2021-12-25','2022-12-25','2023-12-25',
        '2018-01-01', '2019-01-01', '2020-01-01','2021-01-01','2021-01-01','2023-01-01',
        '2018-04-01', '2019-04-21', '2020-04-12', '2021-04-04' ,'2022-04-17','2023-04-09',
        '2018-11-01', '2019-11-01', '2020-11-01', '2021-11-01' ,'2022-11-01','2023-11-01' ])

        st.subheader("📅 Holiday Traffic Trends")
        def holiday_traffic(df):
            summary = df.groupby(['Holiday', 'Vehicle_Date'])['Number_of_Vehicles'].sum().reset_index()
            summary['Type'] = summary['Holiday'].map({True: 'Holiday', False: 'Non-Holiday'})
            fig = px.box(summary, x='Type', y='Number_of_Vehicles',
                                 title='Traffic Volume: Holiday vs Non-Holiday')
            st.plotly_chart(fig, use_container_width=True)

        holiday_traffic(final_df)

    with tab7:
        st.subheader("📆 Traffic Calendar")
        def traffic_calendar(df):
            df['Vehicle_Date'] = pd.to_datetime(df['Vehicle_Date'])
            selected_year = st.selectbox("Select a Year:", sorted(df['Vehicle_Date'].dt.year.unique()),key='year')
            df = df[df['Vehicle_Date'].dt.year == selected_year]

            daily_traffic = df.groupby(df['Vehicle_Date'].dt.date)['Number_of_Vehicles'].sum()
            daily_traffic.index = pd.to_datetime(daily_traffic.index)

            top_10_days = daily_traffic.nlargest(10).index.date
            lt_holidays = holidays.CountryHoliday('LT', years=[selected_year])
            holiday_dates = [pd.to_datetime(d).date() for d in lt_holidays.keys()]

            fig, axs = plt.subplots(3, 4, figsize=(20, 12))
            axs = axs.flatten()

            calendar.setfirstweekday(calendar.MONDAY)
            weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            for i, month in enumerate(range(1, 13)):
                ax = axs[i]
                cal = calendar.monthcalendar(selected_year, month)

                ax.set_title(calendar.month_name[month], fontsize=14)
                ax.set_xlim(0, 7)
                ax.set_ylim(-len(cal)-1, 1)
                ax.axis('off')

                for j, day_name in enumerate(weekdays):
                    ax.text(j + 0.5, 0.5, day_name, ha='center', va='center', fontsize=9, weight='bold')

                for week_num, week in enumerate(cal):
                    for day_idx, day in enumerate(week):
                        if day == 0:
                            continue
                        date_obj = pd.Timestamp(year=selected_year, month=month, day=day).date()
                        count = daily_traffic.get(pd.to_datetime(date_obj), 0)

                        if count > 0:
                            norm_val = min(count / daily_traffic.max(), 0.999)
                            color = sns.color_palette("Blues", 9)[int(norm_val * 8)]
                        else:
                            color = 'white'

                        y_pos = - (week_num + 1)
                        rect = Rectangle((day_idx, y_pos), 1, 1, facecolor=color, edgecolor='gray')
                        ax.add_patch(rect)

                        label = str(day)
                        if date_obj in top_10_days:
                            rect.set_facecolor('orangered')
                        if date_obj in holiday_dates:
                            label += '\nH'

                        ax.text(day_idx + 0.5, y_pos + 0.5, label,
                                ha='center', va='center', fontsize=8, color='white' if date_obj in top_10_days else 'black')

            plt.suptitle(f"Traffic Volume Calendar - {selected_year}", fontsize=20)

            fig.subplots_adjust(bottom=0.1)
            legend_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
            cmap = sns.color_palette("Blues", 9)
            for i, c in enumerate(cmap):
                legend_ax.add_patch(Rectangle((i, 0), 1, 1, color=c))
            legend_ax.set_xlim(0, 9)
            legend_ax.set_xticks([0, 4, 8])
            legend_ax.set_xticklabels(['Low', 'Medium', 'High'])
            legend_ax.set_yticks([])
            legend_ax.set_title('Vehicle Count Intensity (blue)')

            st.pyplot(fig)

            st.markdown("**Legend:**")
            st.markdown("- 🔵 Shades of blue: Number of vehicles (darker = more)")
            st.markdown("- 🔴 Red squares: Top 10 busiest traffic days")
            st.markdown("- 🅷 'H': Lithuanian public holidays")

        traffic_calendar(final_df)

    with tab8:
        st.subheader("🏎️ Speed Behavior Analysis")
        def speed_behavior_plot(df):
            selected_roads = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
            df_filtered_roads = df[df['Road_Number'].isin(selected_roads)].copy()
            allowed_130kmh = ['Car + Trailer / Light Van + Trailer', 'Car / Light Van', 'Car with trailer', 'Cars','Motorcycle', 'Motor Cycle', 'Heavy Van', 'Light Goods', 'Van', 'Bus / Coach','Bus', 'Mini Bus', 'Minibus']
            df_filtered = df_filtered_roads[df_filtered_roads['Vehicle_Type'].isin(allowed_130kmh)].copy()

            if df_filtered.empty:
                st.warning("No data for selected filters.")
                return

            df_filtered['Vehicle_Date'] = pd.to_datetime(df_filtered['Vehicle_Date'])
            selected_year2 = st.selectbox("Select a Year:", sorted(df_filtered['Vehicle_Date'].dt.year.unique()),key='year2')
            df_year = df_filtered[df_filtered['Vehicle_Date'].dt.year == selected_year2].copy()

            def get_speed_limit(date):
                if pd.Timestamp(f"{date.year}-04-01") <= date <= pd.Timestamp(f"{date.year}-10-31"):
                    return 130
                else:
                    return 110

            df_year['Speed_Limit'] = df_year['Vehicle_Date'].apply(get_speed_limit)

            def categorize_speed(row):
                return 'Compliant' if row['Average_Speed'] <= row['Speed_Limit'] else 'Speeding'

            df_year['Speed_Category'] = df_year.apply(categorize_speed, axis=1)

            def get_season(date):
                return 'Summer' if 4 <= date.month <= 10 else 'Winter'

            df_year['Season'] = df_year['Vehicle_Date'].apply(get_season)

            seasonal_analysis = df_year.groupby(['Road_Number', 'Season', 'Speed_Category'])['Number_of_Vehicles'].sum().reset_index()

            fig = px.bar(seasonal_analysis,
                         x='Road_Number',
                         y='Number_of_Vehicles',
                         color='Speed_Category',
                         facet_col='Season',
                         title=f'Seasonal Speed Behavior Analysis on Major Roads for {selected_year2}',
                         labels={'Number_of_Vehicles': 'Number of Vehicles', 'Speed_Category': 'Speeding Behavior'},
                         color_discrete_map={'Compliant': 'green', 'Speeding': 'red'}
                         )
            st.plotly_chart(fig, use_container_width=True)

        speed_behavior_plot(final_df)

    with tab9:
        st.subheader("🛣️ Live Traffic Intensity Map")

        API_URL = "https://eismoinfo.lt/traffic-intensity-service"

        @st.cache_data(ttl=300)
        def fetch_traffic_data():
            response = requests.get(API_URL)
            response.raise_for_status()
            return response.json()

        if st.button("🔄 Refresh Live Data"):
            st.cache_data.clear()

        try:
            data = fetch_traffic_data()
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

        transformer = Transformer.from_crs("EPSG:3346", "EPSG:4326", always_xy=True)

        records = []
        for item in data:
            road_name = item.get('roadName', 'Unknown Road')
            date = item.get('date')
            for segment in item.get('roadSegments', []):
                num_vehicles = segment.get('numberOfVehicles', 0)
                if num_vehicles == 0:
                    continue
                start_x, start_y = segment['startX'], segment['startY']
                end_x, end_y = segment['endX'], segment['endY']
                start_lon, start_lat = transformer.transform(start_x, start_y)
                end_lon, end_lat = transformer.transform(end_x, end_y)
                records.append({
                    'date': date,
                    'numberOfVehicles': num_vehicles,
                    'coordinates': [(start_lon, start_lat), (end_lon, end_lat)],
                    'roadName': road_name,
                    'direction': segment.get('direction', 'N/A'),
                    'averageSpeed': segment.get('averageSpeed', 'unknown'),
                    'trafficType': segment.get('trafficType', 'unknown')
                })

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['date'])

        latest_date = df['date'].dt.date.max()
        filtered_df = df[df['date'].dt.date == latest_date]
        filtered_df = filtered_df[filtered_df['numberOfVehicles'] > 0]


        traffic_colors = {
            'normal': 'green',
            'slow': 'orange',
            'jam': 'red',
            'unknown': 'gray' }

        fig = go.Figure()

        for _, row in filtered_df.iterrows():
            lons, lats = zip(*row['coordinates'])
            traffic_type = row['trafficType'].lower()
            color = traffic_colors.get(traffic_type, 'gray')

            num_vehicles_display = f"{int(row['numberOfVehicles'])}" if row['numberOfVehicles'] > 0 else "unknown"
            avg_speed_display = f"{row['averageSpeed']:.1f} km/h" if row['averageSpeed'] > 0 else "unknown"

            hover_text = (
                f"{row['roadName']}<br>"
                f"Vehicles: {num_vehicles_display}<br>"
                f"Direction: {row['direction']}<br>"
                f"Avg Speed: {avg_speed_display}<br>"
                f"Traffic Type: {row['trafficType']}"
                )

            fig.add_trace(go.Scattermapbox(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(width=4, color=color),
                hoverinfo='text',
                showlegend=False,
                text=hover_text
            ))

        for t_type, color in traffic_colors.items():
            fig.add_trace(go.Scattermapbox(
                lon=[0],
                lat=[0],
                mode='lines',
                line=dict(width=4, color=color),
                name=t_type.capitalize(),
                hoverinfo='skip',
                showlegend=True
            ))

        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                zoom=7,
                center=dict(lat=55.0, lon=24.0)
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(
                title='Traffic Type',
                orientation='h',
                yanchor='bottom',
                y=0.01,
                xanchor='left',
                x=0.01
            )
        )

        st.plotly_chart(fig, use_container_width=True)


    with tab10:
        df = final_df

        st.subheader("🏎️ Predict Average Speed")

        st.markdown("""
            #### Model for Predicting Average Speed

            This model estimates the **average speed** of vehicles on Lithuanian roads based on historical traffic data from 2017 to 2023.
            It uses input features such as **vehicle type**, **road number**, **direction**, **traffic volume**, and **date-related variables** (year, month, weekday).

            By simulating different conditions, users can understand how changes in traffic flow, road segment, or time of year might affect vehicle speeds.
            """)

        vehicle_types = sorted(df["Vehicle_Type"].dropna().unique())
        road_numbers = sorted(df["Road_Number"].dropna().astype(str).unique())
        vehicle_directions = sorted(df["Vehicle_Direction"].dropna().unique())

        vehicle_type_speed = st.selectbox("Vehicle type", vehicle_types, key='speed_vehicletype')
        road_number_speed = st.selectbox("Road number", road_numbers, key='speed_road')
        vehicle_direction_speed = st.selectbox("Vehicle direction (T - in the direction of the road; N - not in the direction of the road)", vehicle_directions, key='speed_direction')
        number_of_vehicles = st.slider("Number of vehicles", 1, 100, 5, key='speed_count')
        year_speed = st.slider("Year", 2017, 2023, 2023, key='speed_year')
        month_speed = st.slider("Month", 1, 12, 5, key='speed_month')
        weekday_speed = st.selectbox("Weekday (1=Mon, 7=Sun)", list(range(1, 8)), key='speed_weekday')

        def create_speed_df():
            return pd.DataFrame([{
                'Vehicle_Type': vehicle_type_speed,
                'Vehicle_Direction': vehicle_direction_speed,
                'Number_of_Vehicles': number_of_vehicles,
                'Road_Number': road_number_speed,
                'Year': year_speed,
                'Month': month_speed,
                'Weekday': weekday_speed
            }])

        if st.button("Predict Speed"):
            try:
                input_df_speed = create_speed_df()
                pred_speed = speed_model.predict(input_df_speed)[0]
                st.success(f"Predicted Average Speed: **{pred_speed:.2f} km/h**")
            except Exception as e:
                st.error(f"Speed prediction error: {e}")

    with tab11:
        df = final_df

        st.subheader("🚚 Predict Traffic Volume")

        st.markdown("""
        #### Model for Predicting Traffic Volume

        This model predicts the **number of vehicles** expected on a road segment given conditions like **average speed**, **vehicle type**, **direction**, **road number**, and **time-based features** (year, month, weekday).
        It is trained on multi-year traffic count data and can help simulate how speed or seasonal changes influence volume.
        """)

        vehicle_types = sorted(df["Vehicle_Type"].dropna().unique())
        road_numbers = sorted(df["Road_Number"].dropna().astype(str).unique())
        vehicle_directions = sorted(df["Vehicle_Direction"].dropna().unique())

        vehicle_type_vol = st.selectbox("Vehicle type", vehicle_types, key='volume_vehicletype')
        road_number_vol = st.selectbox("Road number", road_numbers, key='volume_road')
        vehicle_direction_vol = st.selectbox("Vehicle direction (T - in the direction of the road; N - not in the direction of the road)", vehicle_directions, key='volume_direction')
        average_speed_input = st.slider("Average speed (km/h)", 0, 300, 70, key='volume_speed')
        year_vol = st.slider("Year", 2017, 2023, 2023, key='volume_year')
        month_vol = st.slider("Month", 1, 12, 5, key='volume_month')
        weekday_vol = st.selectbox("Weekday (1=Mon, 7=Sun)", list(range(1, 8)), key='volume_weekday')

        def create_volume_df():
            return pd.DataFrame([{
                'Average_Speed': average_speed_input,
                'Vehicle_Type': vehicle_type_vol,
                'Vehicle_Direction': vehicle_direction_vol,
                'Road_Number': road_number_vol,
                'Year': year_vol,
                'Month': month_vol,
                'Weekday': weekday_vol
            }])

        if st.button("Predict Volume"):
            try:
                input_df_volume = create_volume_df()
                pred_volume = volume_model.predict(input_df_volume)[0]
                st.success(f"Predicted Traffic Volume: **{pred_volume:.0f} vehicles**")
            except Exception as e:
                st.error(f"Volume prediction error: {e}")

    with tab12:
        st.markdown("#### Limitations of this Project")

        st.markdown("""
        - **Incomplete Speed Camera Coverage**
          Not all road segments are equipped with speed detection infrastructure. In some cases, speed cameras were absent, inactive, or malfunctioning, which limited data availability and affected the representativeness of speed-related insights.

        - **Lack of Contextual Environmental Data**
          The dataset did not include weather conditions, road surface quality, or visibility levels—factors that significantly impact traffic behavior. Their absence restricted the development of deeper causal models.

        - **Limited Computing Resources**
          Due to the size of the dataset (over 11 million rows), the Streamlit app encountered memory errors during some runs. This limited the ability to generate more interactive or complex visualizations, especially map-based views.

        - **Geographic Bias in Data Collection**
          Coverage was uneven, with some areas being highly monitored while others lacked adequate sensors. This imbalance may introduce geographic bias in conclusions related to congestion or vehicle flow.

        - **Time Constraints**
          With more time, we would have included animated traffic evolution, predictive modeling, and more detailed filtering options (e.g., by time, weather, road type). Time limits also restricted thorough temporal trend exploration.

        - **Limited Vehicle Classification**
          Vehicle types were grouped broadly. More granular information (e.g., electric vs. combustion engine, passenger vs. commercial) could enhance the specificity of insights.

        - **Absence of Road Condition and Incident Data**
          Key influences like road works or traffic accidents were not included. Their absence hinders accurate attribution of irregularities in traffic speed or volume.

        - **No Real-Time Model Building**
          The dataset continuously updated, making it challenging to build models using the most recent data snapshots. Stable versions of data would help with reproducibility and validation.

        - **High Volume of Missing Data**
          Several columns (e.g., vehicle lane, country on license plates) contained substantial missing values. While these were excluded or imputed where necessary, the overall data sparsity may compromise result reliability or mask underlying patterns.

        - **Exclusion of Zero Speed Values**
          Rows with zero average speed were removed due to ambiguity (non-moving vehicles vs. sensor errors). With more metadata or flags, we could differentiate causes and potentially analyze stopping behavior, road quality effects, or safety patterns using additional data sources like accident reports.

        - **Lack of Rich Vehicle-Specific Attributes**
          Insights could be enhanced with data on vehicle weight, cargo type, or engine size. These variables would support interaction effects in modeling and allow for more detailed traffic or environmental impact analysis.

        - **Unexplained Anomalies in Temporal Trends**
          Some surprising findings, such as the least busy month being February 2018 rather than a COVID-impacted period, suggest that our analysis may have blind spots. This highlights the need for more robust contextual understanding and validation through external sources.
        """)

