"""Some general functions for plotting raster and vector geodata

Index:
make_figure():  Setup fig,axes in the usual way, as a precursor to calling the other functions
zoom_to_data(): Set the boundaries of the axes to the boundaries of the dataframe
choropleth():   Display choropleth of vector data
plot_by_date(): Plot points in input geopandas geodataframe, colored by date 
raster():       Display image of raster data
colorbar():     Make a colorbar for a previously plotted image/choropleth
continuous_to_discrete():   Convert an input *continuous* data array into a discrete set of values

"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from rasterio import plot as rplt
import numpy as np
from numpy import ma 

import raster_utils as ru

def make_figure(shape=1,figsize=None,**kwargs):
    """
    Setup fig,axes in the usual way, as a precursor to calling the other functions

    This is just a shortcut, but instead of calling this you can of course just
    call fig,ax = plt.subplots() to create a figure and axes

    Input:
    shape = (nrows,ncols) in subplots
    figsize = figure size in default matplotlib units
    
    Return: fig,axes = matplotlib.figure.Figure, matplotlib.axes.Axes
    """

    if np.size(shape)==1:
        nrows,ncols = (1,shape)
    elif np.size(shape)==2:
        nrows,ncols = shape
    else:
        print('Invalid shape')
        return

    if figsize is None:
        figsize = (10*ncols, 10*nrows)

    fig,axes = plt.subplots(nrows,ncols,figsize=figsize,**kwargs)

    fig.subplots_adjust(left=0.05,right=0.95,
                        bottom=0.05,top=0.9,
                        wspace=0.05) 
    
    return fig,axes

def zoom_to_data(axes, gdf):
    """
    Set the boundaries of the axes to the boundaries of the dataframe
    """

    aoi_left,aoi_bottom,aoi_right,aoi_top = gdf['geometry'].total_bounds
    axes.set_ylim([aoi_bottom,aoi_top])
    axes.set_xlim([aoi_left,aoi_right])
    return axes

def zoom_to_aoi(axes, aoi_bounds):
    """
    Set the boundaries of the axes to the boundaries of supplied bounds object
    
    Input: 
    axes = matplotlib.axes.Axes object
    aoi_bounds = iterable bounds (left, top, right, bottom) given by
        geopandas geodataframe.total_bounds property, 
        OR BoundingBox given by bounds property of a rasterio dataset  

    """

    aoi_left,aoi_bottom,aoi_right,aoi_top = aoi_bounds
    axes.set_ylim([aoi_bottom,aoi_top])
    axes.set_xlim([aoi_left,aoi_right])
    return axes

def choropleth(axes, gdf,
        colname=None, 
        colormap=None,
        vmin=None,
        vmax=None,
        pmin=0,
        pmax=100,
        title='',
        no_axes_ticks=True,
        use_gpd_plot=True,
        facecolor='None',
        edgecolor='k',
        linewidth=0.5,
        titlesize=15
    ):
    """
    Display choropleth of vector data

    Input:
    axes: a single matplotlib.axes.Axes object
    gdf: a geopandas geodataframe
    colname: column name for plot (if None, show a plain map, using facecolor 
    facecolor: ignored if colname is not None
    edgecolor: color of lines at boundaries of polygons  
    linewidth: width of lines at boundaries of polygons
    colormap: matplotlib colormap used for encoding values of data in colname
              some useful examples include:
              - afmhot
              - gray/Greys
              - viridis
              - inferno2
              - remember append "_r" to reverse the colormap
              https://matplotlib.org/examples/color/colormaps_reference.html
    vmin,vmax = min,max values assigned to 0,255 in the colormap
    pmin,pmax = percentiles used to set vmin,vmax if the latter are None
    
    Return:
    the axes object containing the data (return value of geopandas.plot.show)

    - would be nice to deal with dates
    - should be able to discretise the data? - maybe that's another function
    - should be able to set the vmin,vmax either as kwargs or automatically

    """

    # PROCESS:

    # 0. Initialise/make decisions
    no_choropleth = (colname is None)

    if no_choropleth:
        base = gdf.plot(ax=axes, color=facecolor, edgecolor=edgecolor,
                        linewidth=linewidth)
    else:

        # 1. clean input (NaN, None, numpy.mask)
        valid, = np.where(gdf[colname].values != None)

        # 2. determine vmin/vmax, eg from input or percentiles
        if vmin is None:
            vmin = np.nanpercentile(gdf[colname].astype(float),pmin)
        if vmax is None:
            vmax = np.nanpercentile(gdf[colname].astype(float),pmax)

        # 3. normalise data

        # 4. Make choropleth
        if use_gpd_plot:
            base = gdf.plot(column=colname, ax=axes, 
                                        cmap=colormap,
                                        vmin=vmin,vmax=vmax,
                                        edgecolor='k',linewidth=0.2)


    # 5. set ticks, labels and title
    axes.set_title(title,size=titlesize)

    if no_axes_ticks:
        axes.set_xticks([])
        axes.set_yticks([])

    # 6. return
    if no_choropleth:
        return base
    else:
        return base,(vmin,vmax)


def plot_by_date(axes, gdf, datecolname, 
        colormap=None,
        title='',
        no_axes_ticks=True,
        use_gpd_plot=True,
        pmin=0,
        pmax=100,
        marker='.',
        markersize=5,
        titlesize=20
    ):
    """
    Plot points in input geopandas geodataframe, colored by date 

    Inputs:
    gdf = geopandas geodataframe
    axes = matplotlib.axes.Axes object
    datecolname = column of geodataframe containing dates
    pmin,pmax = percentiles of dates to use to set vmin,vmax

    Returns: axes, 
    (vminf,vmaxf) = vmin,vmax as floats describing min and max of colorscale
    """

    dates = pd.to_datetime(gdf[datecolname])
    valid_dates = (pd.to_numeric(dates).values > 0)

    vminf,vmaxf = np.percentile(pd.to_numeric(dates[valid_dates]),[pmin,pmax])
    vmind,vmaxd = pd.to_datetime([vminf,vmaxf])
    colorscale = dates[valid_dates]

    base = gdf.iloc[valid_dates].plot(ax=axes,
                                      marker=marker,
                                      markersize=markersize,
                                      c=colorscale,
                                      cmap=colormap,
                                      vmin=vmind,
                                      vmax=vmaxd
                                      )

    return base,(vminf,vmaxf)

def raster(axes, im, 
           transform=None,
           colormap=None,
           vmin=None,
           vmax=None,
           pmin=0,
           pmax=100,
           Norm=None,
           rescale_kind='',
           title='',
           no_axes_ticks=True,
           use_rst_plot=True,
           titlesize=15,
           **clahe_equalize_kwargs
    ):
    """
    Display image of raster data
    
    Input:
    axes = matplotlib.axes.Axes object
    im = either image array  [or rasterio dataset??]
    transform = optional Affine transform describing x,y axes of the image
    colormap: matplotlib colormap used for encoding values of data in colname
              some useful examples include:
              - afmhot
              - gray/Greys
              - viridis
              - inferno2
              - remember append "_r" to reverse the colormap
              https://matplotlib.org/examples/color/colormaps_reference.html
    vmin,vmax = min,max values assigned to 0,255 in the colormap
    pmin,pmax = percentiles used to set vmin,vmax if the latter are None
    rescale_kind = string to pass to raster_utils.rescale_image as 'kind'
        Note that using rescale_kind, the colorbar may not be labelled 
        correctly, hence it is better to pass Norm
    Norm = an instance of one of the normalization classes from 
        matplotlib.colors, as an alternative to passing rescale_kind
        If neither of these are passed then a simple linear normalization
        between vmin and vmax is used

    use_rst_plot = whether to use rasterio plot (otherwise just pyplot.imshow)
    

    Return:
    the axes object containing the data (return value of rasterio.plot.show
        or pyplot.imshow)
    the Normalize instance of the axes

    """

    # PROCESS:

    # 1. clean input (NaN, None, numpy.mask)
    data = np.squeeze(im).copy()
    Nans = np.array(~np.isfinite(data))
    Nones = np.array(data is None)
    datamask = (Nans | Nones)
    data = ma.array(data,mask=datamask,fill_value=np.nan).filled()

    # 2. determine vmin/vmax, eg from input or percentiles
    if vmin is None:
        vmin = np.nanpercentile(data,pmin)
    if vmax is None:
        vmax = np.nanpercentile(data,pmax)

    data = np.where(data>=vmin,data,vmin)
    data = np.where(data<=vmax,data,vmax)

    # 3. normalise data
    # There are three options:
    # If rescale_kind is passed as a string, rescale_image will be used
    # Otherwise, can use matplotlib.colors to normalize the data - the
    # Norm instance can be passed as an input 
    # If neither is passed then a simple linear normalization is used
    if len(rescale_kind)>0:
        data = ru.rescale_image(data,kind=rescale_kind,
                                **clahe_equalize_kwargs)
    #
    if Norm is None:
        Norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)

    # 4. display
    if use_rst_plot:
        base = rplt.show(data,ax=axes,cmap=colormap,vmin=vmin,vmax=vmax,
                  title=title,transform=transform,
                  norm=Norm
        )
    else:
        base = axes.imshow(data,cmap=colormap,vmin=vmin,vmax=vmax,
                           norm=Norm)
        # base = axes.pcolormesh(data,norm=Norm,cmap=colormap)
        # Norm = vmin,vmax

    # 5. set ticks, labels and title
    axes.set_title(title,size=titlesize)
    
    if no_axes_ticks:
        axes.set_xticks([])
        axes.set_yticks([])

    # 6. return
    return base,Norm #(vmin,vmax)


def colorbar(axes,Norm,
        colormap=None,
        discrete=False,
        disc_bounds=[],
        disc_tickvals=[],
        disc_ticklabs=[],
        isdate=False,
        label='',
        ticks=4,
        locbottom=False
    ):
    """
    Make a colorbar for a previously plotted image/choropleth
    
    Input:
    axes = matplotlib.axes.Axes object
    Norm = an instance of one of the Normalize classes in matplotlib.colors
    OR Norm = tuple (vmin,vmax) of data plotted in axes (assumed linear scale)

    - has to be able to deal with dates as well as float data
    - should be able to specify ticks and ticklabels
    - should be compatible with data plotted as raster or vector
    - should be able to plot a discretised colorbar from a continuous cmap
    - either plot across full width/height of whole figure, or across a single
      axes
    - would be nice if it could also locate the axes as an inset within the
      main image

    """

    # PROCESS:

    # 1. position the colorbar
    #cbpos = [0.05,0.95,0.9,0.02]
    if (type(axes)==matplotlib.image.AxesImage) or (type(axes)==matplotlib.collections.QuadMesh):
        Aaxes=axes.axes
    else:
        Aaxes=axes

    if locbottom:
        cbpos = [Aaxes.get_position().xmin,0.01,Aaxes.get_position().width,0.02]
    else:
        cbpos = [Aaxes.get_position().xmin,0.97,Aaxes.get_position().width,0.02]


    # 2. determine vmin, vmax 
    # vmin,vmax = vminmax

    # Ensure backwards compatibility with version where vmin,vmax were passed
    if type(Norm)==tuple:
        vmin,vmax = Norm
        Norm = plt.Normalize(vmin=vmin,vmax=vmax)

    # 3. define discrete bounds if required 

    # 4. plot colorbar
    fig = axes.figure
    cax = fig.add_axes(cbpos,frameon=True,clip_on=True)

    if discrete:
        CB0 = matplotlib.colorbar.ColorbarBase(cax,orientation='horizontal',
                                              boundaries= disc_bounds,
                                              cmap=colormap,
                                              norm=Norm
                                              # norm=plt.Normalize(vmin=vmin, vmax=vmax)
                                              )
        CB0.set_ticks(disc_tickvals)
        CB0.set_ticklabels(disc_ticklabs)
    else:
        sm = plt.cm.ScalarMappable(cmap=colormap, 
                               norm=Norm
                               # norm=plt.Normalize(vmin=vmin, vmax=vmax)
                               )
        sm._A = []
        CB0 = fig.colorbar(sm, orientation='horizontal', cax=cax)
        
        # 5. set ticks, labels 
        if isdate:
            # Beware of the following line when using any Norm other than linear Normalize()
            cax.xaxis.set_major_locator(plt.MaxNLocator(ticks+1))
            cax_date_labels = pd.to_datetime(cax.xaxis.get_ticklocs()).strftime('%Y-%m-%d')
            cax.xaxis.set_ticklabels(cax_date_labels)

    # 5. set title
    CB0.set_label(label)
    
    
    # 6. return axes
    return CB0


def continuous_to_discrete(cont_data, bounds, values=[], labels=[]):
    """
    Convert an input *continuous* data array into a discrete set of values

    Input:
    vector array or image with continuous scale
    set of boundaries and labels to go with them
    bounds = N+1 bounds to be applied to input data to separate N discrete classses
    labels = list of N labels for the discrete classes
    values = list of N values to assign to the discrete classes:
        if values is None then it is set to the mid-points

    Output:
    discretized array with each pixel assigned to discrete classes 
        given by input bounds
    tick labels and tick values to be passed to colorbar

    
    """

    
    N = len(bounds)-1
    disc_data = cont_data.copy()
    assign_values = []

    for ii in range(N):
        if len(values)==N:
            assign_val = values[ii]
        else:
            assign_val = (bounds[ii]+bounds[ii+1])/2.0
        
        assign_values = assign_values+[assign_val]
        disc_data[(disc_data>=bounds[ii]) & (disc_data<bounds[ii+1])] = assign_val

    # Set colorbar ticks and labels
    CB_tickvalues = np.array(list(zip(bounds,assign_values))).flatten()
    
    if len(labels)==N:
        CB_ticklabels = np.array(list(zip(['']*N,labels))).flatten()
    else:
        #CB_ticklabels = np.array(['']*(N*2)).flatten()
        assign_values_str = [str(a) for a in assign_values]
        CB_ticklabels = np.array(list(zip(['']*N,assign_values_str))).flatten()

    # CB0.set_ticks(CB_tickvalues)
    # CB0.set_ticklabels(CB_ticklabels)
    
    return disc_data, CB_tickvalues, CB_ticklabels


# ======================
