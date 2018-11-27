"""Some general functions for plotting raster and vector geodata

Before calling these functions, a pyplot Figure and Axes pair should have 
been created using e.g. fig,ax = plt.subplots()
"""

import matplotlib
import matplotlib.pyplot as plt

def choropleth(gdf,axes,
        colname=None, 
        colormap=None,
        title='',
        no_axes_ticks=True,
        use_gpd_plot=True,
        facecolor=None,
        edgecolor=None,
        linewidth=None,
        titlesize=20
    ):
    """
    Display choropleth of vector data

    Input:
    gdf: a geopandas geodataframe
    axes: a single matplotlib.axes.Axes object
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

        # 3. normalise data

        # 4. Make choropleth
        if use_gpd_plot:
            base = gdf.plot(column=colname, ax=axes, 
                                        cmap=colormap,
                                        scheme='equal_interval',
                                        edgecolor='k',linewidth=0.2)


        # 5. set ticks, labels and title
        axes.set_title(title,size=titlesize)

        if no_axes_ticks:
            axes.set_xticks([])
            axes.set_yticks([])

    # 6. return
    return base



def raster():
    """
    Display image of raster data
    
    Input:
    im/src: either image array or rasterio dataset?
    axes:
    colormap:
    vmin, vmax:
    normalise method or matplotlib Norm object?:
    Affine transform describing x,y axes of the image?

    Return:
    the axes object containing the data (return value of rasterio.plot.show
        or pyplot.imshow)
    

    """

    # PROCESS:

    # 1. clean input (NaN, None, numpy.mask)
    # Nans = np.array(~np.isfinite(inputdata))
    # Nones = np.array(inputdata is None)
    # data = ma.array(inputdata,mask=(Nans | Nones))

    # 2. determine vmin/vmax, eg from input or percentiles
    if pmin>=0 and pmax<=100:
        vmin,vmax = np.nanpercentile(data,[pmin,pmax])

    # 3. normalise data
    # data = ru.rescale_image(inputdata,kind='linear')

    # 4. display
    if use_rst_plot:
        base = rplt.show(data,ax=axes,vmin=vmin,vmax=vmax,cmap=colormap,
                  title=title)
    else:
        base = axes.imshow(data,vmin=vmin,vmax=vmax,cmap=colormap)


    # 5. set ticks, labels and title

    fig.subplots_adjust(left=0.05,right=0.95,
                        bottom=0.05,top=1,
                        wspace=0.05) 

    axes.set_title(title)
    
    if no_axes_ticks:
        axes.set_xticks([])
        axes.set_yticks([])

    # 6. return
    return base

def colorbar():
    """
    Make a colorbar for a previously plotted image/choropleth
    
    Input:

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
    cbpos = [0.05,0.95,0.9,0.02]
    cbpos = [axes.get_position().xmin,0.97,axes[0].get_position().width,0.02]


    
    # adjust alignment of subplots...
    fig.subplots_adjust(left=0.05,right=0.95,
                        bottom=0.05,top=1,
                        wspace=0.05) 

 

    # 2. determine vmin, vmax and cmap

    # 3. define discrete bounds if required 

    # 4. plot colorbar

    cax=fig.add_axes(cbpos,frameon=True,clip_on=True)

    if discrete:
        CB0 = matplotlib.colorbar.ColorbarBase(cax,orientation='horizontal',
                                              boundaries= disc_bounds,
                                              cmap=cmap,norm=base.norm)
        CB0.set_ticks(disc_tickvals)
        CB0.set_ticklabels(disc_ticklabs)
    else:
        sm = plt.cm.ScalarMappable(cmap=colormap, 
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        CB0 = fig.colorbar(sm, orientation='horizontal', cax=cax)


    # 5. set ticks, labels and title
    CB0.set_label(label)
    
    if isdate:
        cax.xaxis.set_major_locator(plt.MaxNLocator(4))
        cax_date_labels = pd.to_datetime(cax.xaxis.get_ticklocs()).strftime('%Y-%m-%d')
        cax.xaxis.set_ticklabels(cax_date_labels)

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
        disc_data[disc_data>=bounds[ii] & disc_data<bounds[ii+1]] = assign_val

    # Set colorbar ticks and labels
    CB_tickvalues = np.array(list(zip(bounds,assign_values))).flatten()
    
    if len(labels)==N:
        CB_ticklabels = np.array(list(zip(['']*N,labels))).flatten()
    else:
        CB_ticklabels = np.array(['']*(N*2)).flatten()

    # CB0.set_ticks(CB_tickvalues)
    # CB0.set_ticklabels(CB_ticklabels)
    
    return disc_data, CB_tickvalues, CB_ticklabels

def dates_to_continuous():
    """
    Convert input pandas datetime series to continuous float series
    and output vmin,vmax as well as tick parameters for colorbar



    """

# ======================
