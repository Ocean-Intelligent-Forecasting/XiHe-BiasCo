This is the official repository for the XiHe-BiasCo papers.

Physically Coherent Bias Correction of Operational Numerical Global Ocean Forecasting via Multivariable Knowledge Transferg, 2026.

*by Xiang Wang, Qingye Min, Zhipan Li, Junxing Zhu, Guihua Wang, Huizan Wang, Yi Han, Guang Yu, Hongze Leng, Kefeng Deng, Junqiang Song* 

Resources including pre-trained models, and inference code are released here.




### Input data

- `input_surface_data` stores the input data for 1 to 22 layers. It is a NumPy array shaped **(1,52,2041,4320)** which represents the variables **(Time, Variables, Lat, Lon)**. The Variables are organized in the following order: **zos** (sea surface height above geoid), **u** (era5 10m zonal component of sea surface wind), **v** (era5 10m meridional component of sea surface wind), **sst** (sea surface temperature), **thetao_0, so_0, uo_0, vo_0,..., thetao_21, so_21, uo_21, vo_21** (the 1st layer of ocean temperature, the 1st layer of ocean salinity, the zonal component of the 1st layer of ocean currents, the meridional component of the 1st layer of ocean currents, ..., the 22nd layer of ocean temperature, the 22nd layer of ocean salinity, the zonal component of the 22nd layer of ocean currents, the meridional component of the 22nd layer of ocean currents).

- `input_deep_data` stores the input data for 23 to 33 layers. It is a NumPy array shaped **(1,48,2041,4320)** which represents the variables **(Time, Variables, Lat, Lon)**. The Variables are organized in the following order: **zos** (sea surface height above geoid), **u** (era5 10m zonal component of sea surface wind), **v** (10m meridional component of sea surface wind), **sst** (sea surface temperature), **thetao_22, so_22, uo_22, vo_22,..., thetao_32, so_32, uo_32, vo_32** (the 23rd layer of ocean temperature, the 23rd layer of ocean salinity, the zonal component of the 23rd layer of ocean currents, the meridional component of the 23rd layer of ocean currents, ..., the 33rd layer of ocean temperature, the 33rd layer of ocean salinity, the zonal component of the 33rd layer of ocean currents, the meridional component of the 33rd layer of ocean currents).

> In both cases, the dimensions of 2041 and 4320 represent the size along the latitude and longitude, where the numerical range is [-80,90] degree and [-180,180] degree, respectively, and the spacing is 1/12 degrees. For each 2041x4320 slice, the data format is exactly the same as the `.nc` file download from the [PSY4 Forecasts](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/services) official website.
>
> Note that the NumPy arrays should be in single precision (`.astype(np.float32)`).

### Output data

The model predicts 6 ocean variables. There are 23 layers including: **ocean temperature, salinity, zonal and meridional components of ocean current** (i.e., layer 1: 0.49m, layer 3: 2.65m, layer 5: 5.08m, layer 7: 7.93m, layer 9: 11.41m, layer 11: 15.81m, layer 13: 21.60m, layer 15: 29.44m, layer 17: 40.34m, layer 19: 55.76m, layer 21: 77.85m, layer 22: 92.32m, layer 23: 109.73m, layer 23: 109.73m. The 24th layer: 130.67m, the 25th layer: 155.85m, the 26th layer: 186.13m, the 27th layer: 222.48m, the 28th layer: 266.04m, the 29th layer: 318.31m, the 30th layer: 380.21m, the 31st layer: 453.94m, the 32nd layer: 541.09m and the 33rd layer: 643.57m)，**sea surface height above geoid** and **sea surface temperature**.
