from matplotlib.collections import PatchCollection
from matplotlib.colors import from_levels_and_colors
from matplotlib.patches import Polygon, Patch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from .smopy import Map as SmopyMap
from sklearn.preprocessing import minmax_scale

def feature_to_patch(s, smopy_map):
    s_arr = np.array(s.exterior.xy).T
    s_pos = np.array(smopy_map.to_pixels(s_arr[:,1], s_arr[:,0])).T
    return Polygon(s_pos, closed=False)

def choropleth(geodf, figsize=12, column=None, scheme='fisher_jenks', 
                   n_colors=5, palette='viridis', alpha=0.75, cbar_orientation='vertical'):
        
    bounds = geodf.total_bounds
    bbox = (bounds[1], bounds[0], bounds[3], bounds[2])
    smopy_map = SmopyMap(bbox, z=12, margin=0)

    fig_shape = smopy_map.to_numpy().shape
    aspect = fig_shape[0] / fig_shape[1]

    fig = plt.figure(figsize=(figsize, figsize / aspect))
    plt.imshow(smopy_map.img)
    ax = plt.gca()
    plt.axis('off')
    
    choro = []
    patch_values = []
    
    if scheme != 'categorical':
        
        for idx, row in geodf.iterrows():
            feature = row.geometry
            value = row[column]
            
            if feature.geom_type == 'Polygon':
                choro.append(feature_to_patch(feature, smopy_map))
                patch_values.append(value)
            elif feature.geom_type == 'MultiPolygon':
                for subfeature in feature:
                    choro.append(feature_to_patch(subfeature, smopy_map))
                    patch_values.append(value)
            else:
                continue
        
        binning = gpd.plotting.__pysal_choro(geodf[column], scheme='fisher_jenks', k=n_colors)
        bins = np.insert(binning.bins, 0, geodf[column].min())
        palette_values = sns.color_palette(palette, n_colors=n_colors)
        cmap, norm = from_levels_and_colors(bins, palette_values, extend='neither')
        cmap.set_over(palette_values[-1], alpha=alpha)

        collection = PatchCollection(choro, alpha=alpha, cmap=cmap, norm=norm)    
        collection.set_array(np.array(patch_values))

        
        
        if cbar_orientation is not None:
            plt.colorbar(collection, shrink=0.5, orientation='vertical', label=column, 
                     fraction=0.05, pad=0.01)
    else:
        category_values = sorted(geodf[column].unique())
        n_colors = len(category_values)
        palette = sns.color_palette(palette, n_colors=n_colors)
        color_dict = dict(zip(category_values, palette))
        
        
        for idx, row in geodf.iterrows():
            feature = row.geometry
            value = row[column]
            
            if feature.geom_type == 'Polygon':
                choro.append(feature_to_patch(feature, smopy_map))
                patch_values.append(color_dict[value])
            elif feature.geom_type == 'MultiPolygon':
                for subfeature in feature:
                    choro.append(feature_to_patch(subfeature, smopy_map))
                    patch_values.append(color_dict[value])
            else:
                continue

        collection = PatchCollection(choro, alpha=alpha, color=patch_values)   
        
        bbox_to_anchor = None#(0.99, 0.75)
        legend_parts = [Patch(color=color, label=label) for label, color in zip(category_values, palette)]
        plt.legend(legend_parts, [p.get_label() for p in legend_parts], bbox_to_anchor=bbox_to_anchor)
        
    ax.add_collection(collection)
    plt.tight_layout()
    
    return ax
    

def markers(geodf, figsize=12, column=None, n_colors=5, palette='viridis', alpha=0.75,
                min_size=1, max_size=30, color='purple', scheme=None):
    '''Display a plot map of points in a GeoDataFrame.
    
Parameters
----------
geodf: GeoDataFrame of Point 
column: size of the makers. None by default.

    
    '''
    #bbox = (-33.67908699999886, -70.83500200000343, -33.313337999998865, -70.46741200000102)
    
    bounds = geodf.total_bounds
    bbox = (bounds[1], bounds[0], bounds[3], bounds[2])
    smopy_map = SmopyMap(bbox, z=12, margin=0)
    
    fig_shape = smopy_map.to_numpy().shape
    aspect = fig_shape[0] / fig_shape[1]
    
    fig = plt.figure(figsize=(figsize, figsize / aspect))
    plt.imshow(smopy_map.img, interpolation='bicubic')
    ax = plt.gca()
    plt.axis('off')

    if scheme == 'categorical':
        category_values = sorted(geodf[column].unique())
        n_colors = len(category_values)
        palette = sns.color_palette(palette, n_colors=n_colors)
        color_dict = dict(zip(category_values, palette))

        for cat in category_values:
            cat_df = geodf[geodf[column] == cat]
            x, y = smopy_map.to_pixels(cat_df.geometry.y, cat_df.geometry.x)
            # TODO: size scale
            ax.scatter(x, y, s=min_size, color=color_dict[cat], label=cat)

        ax.legend()
    else:
        x, y = smopy_map.to_pixels(geodf.geometry.y, geodf.geometry.x)
        #choro = geodf.geometry.map(lambda x: feature_to_patch(x, smopy_map))
    
        if column is not None:
            sizes = minmax_scale(geodf[column], feature_range=(min_size, max_size))
            ax.scatter(x, y, s=sizes, color=color)
        else:
            ax.scatter(x, y, color=color)
        
    return ax


def plot_lines(geodf, figsize=12, linewidth=1, color_column=None, alpha=0.75, zorder=1):
    from matplotlib.collections import LineCollection

    bounds = geodf.total_bounds
    bbox = (bounds[1], bounds[0], bounds[3], bounds[2])
    smopy_map = SmopyMap(bbox, z=12, margin=0)
    
    fig_shape = smopy_map.to_numpy().shape
    aspect = fig_shape[0] / fig_shape[1]
    
    fig = plt.figure(figsize=(figsize, figsize / aspect))
    plt.imshow(smopy_map.img)
    ax = plt.gca()

    kwargs = {'alpha':alpha, 'linewidth':linewidth, 'zorder':zorder}
    if color_column is not None:
        kwargs['color'] = geodf[color_column]
    
    segments = [np.array(linestring)[:, :2] for linestring in geodf.geometry]
    segments = [smopy_map.to_pixels(line[:,[1,0]]) for line in segments]
    collection = LineCollection(segments, **kwargs)
    
    ax.add_collection(collection, autolim=False)
    ax.axis('off')


def markers_layers(geodf, figsize=12, column=None, n_colors=5, palette='viridis', alpha=0.75,
                min_size=1, max_size=30, color='purple', 
                layer=None, layer_linewidth=1, layer_alpha=0.75, layer_color_col=None):
    '''Display a plot map of points in a GeoDataFrame.
    
Parameters
----------
geodf: GeoDataFrame of Point 
column: size of the makers. None by default.

    
    '''
    #bbox = (-33.67908699999886, -70.83500200000343, -33.313337999998865, -70.46741200000102)
    
    bounds = geodf.total_bounds
    bbox = (bounds[1], bounds[0], bounds[3], bounds[2])
    smopy_map = SmopyMap(bbox, z=12, margin=0)
    
    fig_shape = smopy_map.to_numpy().shape
    aspect = fig_shape[0] / fig_shape[1]
    
    fig = plt.figure(figsize=(figsize, figsize / aspect))
    plt.imshow(smopy_map.img)
    ax = plt.gca()
    #plt.xlim(0, smopy_map.w)
    #plt.ylim(smopy_map.h, 0)
    plt.axis('off')
    
    x, y = smopy_map.to_pixels(geodf.geometry.y, geodf.geometry.x)
    #choro = geodf.geometry.map(lambda x: feature_to_patch(x, smopy_map))
    
    if layer is not None:
        geom_types = layer.geometry.type
        line_idx = np.asarray((geom_types == 'LineString') #| (geom_types == 'MultiLineString')
                             )
        layer_lines = layer[line_idx]
        if not layer_lines.empty:
            plot_lines(smopy_map, ax, layer_lines, layer_linewidth, layer_color_col, layer_alpha, zorder=10)
    
    if column is not None:
        sizes = minmax_scale(geodf[column], feature_range=(min_size, max_size))
        ax.scatter(x, y, s=sizes, color=color, zorder=20)
    else:
        ax.scatter(x, y, color=color, zorder=20)
        
    return ax

def markers2(geodf, figsize=12, column=None, n_colors=5, palette='viridis', alpha=0.75,
                min_size=1, max_size=30, color='purple'):
    '''Display a plot map of points in a GeoDataFrame.
    
Parameters
----------
geodf: GeoDataFrame of Point 
column: size of the makers. None by default.

    
    '''
    #bbox = (-33.67908699999886, -70.83500200000343, -33.313337999998865, -70.46741200000102)
    
    bounds = geodf.total_bounds
    bbox = (bounds[1], bounds[0], bounds[3], bounds[2])
    smopy_map = SmopyMap(bbox, z=12, margin=0)
    
    fig_shape = smopy_map.to_numpy().shape
    aspect = fig_shape[0] / fig_shape[1]
    
    fig = plt.figure(figsize=(figsize, figsize / aspect))
    plt.imshow(smopy_map.img)
    ax = plt.gca()
    plt.axis('off')
    
    x, y = smopy_map.to_pixels(geodf.geometry.y, geodf.geometry.x)
    #choro = geodf.geometry.map(lambda x: feature_to_patch(x, smopy_map))
    
    if column is not None:
        sizes = minmax_scale(np.abs(geodf[column]), feature_range=(min_size, max_size))
        
        cm = plt.cm.get_cmap(palette)
        sc = ax.scatter(x, y, s=sizes, c=geodf[column], cmap=cm)
        fig.colorbar(sc, ax=ax)
        
        #ax.scatter(x, y, s=sizes, color=color)
    else:
        ax.scatter(x, y, color=color)
    
    return ax
