import copy
import numpy as np
import matplotlib.pyplot as plt

def plot_vector_lattice_schematic(ax,pattern,L=10,fontsize_multiplier=1,grid_multiplier=0.8):
    """
    Plots the data organized in an LxL lattice with explicit 
    vector notation (sigma_i) and index labels.
    """
    pattern = np.atleast_1d(pattern).reshape((L,L))
    # We use a slight offset to draw "boxes" rather than just an image
    ax.text(-0.5, L - 1 - 0.5, r'$\vec{\sigma}=\{$', 
            color='k', fontsize=36*fontsize_multiplier, ha='right', va='bottom', alpha=1, weight='normal')
    for y in range(L):
        for x in range(L):
            # Index formula: 1 + x + L*y (as requested)
            # Note: in plotting, y-axis usually starts from top, 
            # so we use (L - 1 - y) if you want index 1 at the top-left
            idx = 1 + x + L * y
            
            # Determine color from the pattern
            pixel_val = pattern[y, x]
            face_col = 'black' if pixel_val < 0 else 'white'
            edge_col = 'gray'
            text_col = 'white' if pixel_val < 0 else 'black'
            
            # Draw the neuron as a square box
            rect = plt.Rectangle((x - 0.5, L - 1 - y - 0.5), grid_multiplier*1, grid_multiplier*1, 
                                 facecolor=face_col, edgecolor=(0,0,0,0.8), linewidth=0.5)
            ax.add_patch(rect)
            if (x+1)*(y+1) < (L**2-1):
                ax.text(x + 0.5, L - 1 - y - 0.5, ',', 
                        color='k', fontsize=34*fontsize_multiplier, ha='right', va='bottom', alpha=1, weight='normal')
            ax.text(x-0.15, L - 1 - y-0.15, f"$\sigma_{{{idx}}}$", 
                    color=text_col, fontsize=20*fontsize_multiplier, fontweight='bold', ha='center', va='center')
            
    ax.text((L-1) + 0.8, L - 1 - (L-1) - 0.5, r'$\}$', 
                color='k', fontsize=36*fontsize_multiplier, ha='right', va='bottom', alpha=1, weight='normal')
    rect1 = plt.Rectangle((-0.5, L - 1 - (L-1) - 2.5), grid_multiplier*1, grid_multiplier*1, 
                            facecolor='w', edgecolor=(0,0,0,0.8), linewidth=0.5, clip_on=False)
    rect2 = plt.Rectangle((-0.5, L - 1 - (L-1) - 3.5), grid_multiplier*1, grid_multiplier*1, 
                            facecolor='k', edgecolor=(0,0,0,0.8), linewidth=0.5, clip_on=False)
    ax.text(2.0, L - 1 - (L-1)-0.15-2.0, "$\sigma_{i}=+1$", color='k', fontsize=28*fontsize_multiplier, fontweight='bold', ha='center', va='center')
    ax.text(2.0, L - 1 - (L-1)-0.15-3.0, "$\sigma_{i}=-1$", color='k', fontsize=28*fontsize_multiplier, fontweight='bold', ha='center', va='center')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    # Formatting the plot
    ax.set_xlim(-1, L)
    ax.set_ylim(-1, L)
    ax.set_aspect('equal')
    ax.axis('off')


def _set_default_kwargs(kwargs_dict,**default_args):
    """
    kwargs_dict is the '**kwargs' argument of any function
    this function checks whether any argument in kwargs_dict has a default value given in default_args...
    if yes, and the corresponding default_args key is not in kwargs_dict, then includes it there

    this is useful to avoid duplicate key errors
    """
    kwargs_dict = copy.deepcopy(kwargs_dict)
    for k,v in default_args.items():
        if not (k in kwargs_dict):
            kwargs_dict[k] = v
    return kwargs_dict

def _exists(X):
    return type(X) is not type(None)

def _get_kwargs(args,**defaults):
    args = args if _exists(args) else dict()
    return _set_default_kwargs(args,**defaults)


def plot_overlap_evolution(ax, m_array, colors=None, t_scale=1.0, labels=None, t_label=None, label_args=None, line_args=None):
    """
    Plots the evolution of the overlap (magnetization) over iterations.
    
    Parameters:
    - m_array: numpy array of shape (P, T) where P is patterns and T is iterations.
    - labels: list of strings for the legend.
    """
    P, T       = m_array.shape
    iterations = t_scale*np.arange(T)
    t_label    = np.atleast_1d(T-1    if type(t_label) is type(None) else t_label)
    labels     = np.atleast_1d(''     if type(labels)  is type(None) else labels)
    colors     = plt.cm.tab10.colors  if type(colors)  is type(None) else colors
    t_label    = np.tile(t_label, max((1,P-t_label.size+1)))[:P]
    labels     = np.tile(labels , max((1,P-labels.size+1 )))[:P]
    n_colors   = np.atleast_2d(colors).shape[0]
    ax.axhline( 1.0, clip_on=False,color='k', lw=1.0, linestyle='--', alpha=0.8, label='_Attractor Limit')
    ax.axhline(-1.0, clip_on=False,color='k', lw=1.0, linestyle='--', alpha=0.8, label='_Attractor Limit')
    for mu in range(P):
        label = labels[mu] if len(labels[mu]) else f"Pattern $\\mu={mu+1}$"
        ax.plot(iterations, m_array[mu, :],**_get_kwargs(line_args,clip_on=False, marker='none', markersize=5, linewidth=2, label=label, color=colors[mu % n_colors]))
        if labels[mu]:
            ax.text(iterations[t_label[mu]],m_array[mu,t_label[mu]],labels[mu],**_get_kwargs(label_args,ha='right',va='top',color=colors[mu % n_colors],fontsize=10,clip_on=False))
        
    # Standard Hopfield bounds: 1.0 (Full Memory), -1.0 (Inverted Memory)
    ax.set_ylim(-1.01, 1.01)
    ax.set_xlim(0, t_scale*(T-1))
    ax.set_yticks([-1,1])
    [ax.spines[s].set_visible(False) for s in ['top', 'right', 'bottom']]
          
    #plt.tight_layout()
    return ax

def enlarge_axis(ax,mult=1,dx=0.0,dy=0.0,mx=None,my=None):
    mx = mx if mx else mult
    my = my if my else mult
    for a in np.atleast_1d(ax).flatten():
        a.set_position([box.x0 + ((1-mx)/2)*box.width+dx, 
                        box.y0 + ((1-my)/2)*box.height+dy,
                        box.width * mx,
                        box.height * my]) if (box := a.get_position()) else None
    return ax

def unset_minor_grid(ax):
    for a in np.atleast_1d(ax).flatten():
        a.grid(False,which='minor')
    return ax

def shift_axes_title(ax,dx=0,dy=0):
    for a in np.atleast_1d(ax).flatten():
        a.title.set_position([a.title.get_position()[0] + dx, a.title.get_position()[1] + dy])
    return ax