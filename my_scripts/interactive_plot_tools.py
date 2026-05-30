# interactive_plot_tools.py
# A file for interactively edit the plots to add insets, boundaries etc

"""
interactive_plot_tools.py

Shared interactive plotting utilities for solvation shell analysis.
Provides reusable tools for interactive plot manipulation across multiple classes.

Author: R.Swai
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def interactive_rectangle_selector(ax, initial_bbox=None, title="Interactive Rectangle Editor"):
    '''
    Generic interactive rectangle selector that works on any matplotlib axes.
    Click near corners/edges to resize, click center to move.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the rectangle to
    initial_bbox : tuple, optional
        Initial rectangle position (x, y, width, height) in data coordinates
        If None, places rectangle at 60% x-position, 65% y-position with 25% dimensions
    title : str
        Title to add to the plot
    
    Returns
    -------
    rect_params : dict
        Dictionary with inset parameters:
        - 'xlim': (xmin, xmax) tuple
        - 'ylim': (ymin, ymax) tuple
        - 'bbox': [xmin, xmax, ymin, ymax] list
    
    Usage
    -----
    # In your plotting method:
    fig, ax = plt.subplots()
    ax.plot(x, y)  # Your data
    
    params = interactive_rectangle_selector(ax, title="Select Inset Region")
    
    # Use params in your inset:
    inset_xlim = params['xlim']
    inset_ylim = params['ylim']
    '''
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Initial rectangle
    if initial_bbox is None:
        rect_width = (xlim[1] - xlim[0]) * 0.25
        rect_height = (ylim[1] - ylim[0]) * 0.25
        rect_x = xlim[0] + (xlim[1] - xlim[0]) * 0.60
        rect_y = ylim[0] + (ylim[1] - ylim[0]) * 0.65
    else:
        rect_x, rect_y, rect_width, rect_height = initial_bbox
    
    rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                    linewidth=2, edgecolor='red', facecolor='none',
                    linestyle='--', alpha=0.8)
    ax.add_patch(rect)
    
    # Store state
    state = {
        'dragging': False,
        'resize_mode': None,
        'start_pos': None,
        'rect': rect,
        'initial_rect_pos': None,
        'initial_rect_size': None
    }
    
    # Text annotation
    text_bbox = dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8)
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=text_bbox)
    
    # Update title
    original_title = ax.get_title()
    ax.set_title(f'{title}\nCLICK CORNERS/EDGES to RESIZE | CLICK CENTER to MOVE | ENTER to save | ESC to cancel',
                fontsize=12, fontweight='bold')
    
    def update_info_text():
        x, y = rect.get_xy()
        w, h = rect.get_width(), rect.get_height()
        mode = f"RESIZE ({state['resize_mode']})" if state['resize_mode'] else "MOVE"
        status = f"[{mode}]" if state['dragging'] else ""
        
        info = (f'Inset Box {status}:\n'
               f'X: [{x:.1f}, {x+w:.1f}]\n'
               f'Y: [{y:.5f}, {y+h:.5f}]\n'
               f'\nCLICK corners/edges: Resize\n'
               f'CLICK center: Move\n'
               f'ENTER: Save | ESC: Cancel')
        info_text.set_text(info)
    
    update_info_text()
    
    def get_resize_mode(click_x, click_y, rect_x, rect_y, rect_w, rect_h):
        '''Determine resize mode based on click position'''
        edge_threshold_x = rect_w * 0.15
        edge_threshold_y = rect_h * 0.15
        
        near_left = abs(click_x - rect_x) < edge_threshold_x
        near_right = abs(click_x - (rect_x + rect_w)) < edge_threshold_x
        near_bottom = abs(click_y - rect_y) < edge_threshold_y
        near_top = abs(click_y - (rect_y + rect_h)) < edge_threshold_y
        
        # Corner modes
        if near_right and near_top:
            return 'ne'
        elif near_right and near_bottom:
            return 'se'
        elif near_left and near_top:
            return 'nw'
        elif near_left and near_bottom:
            return 'sw'
        # Edge modes
        elif near_right:
            return 'e'
        elif near_left:
            return 'w'
        elif near_top:
            return 'n'
        elif near_bottom:
            return 's'
        
        return None  # Center (move mode)
    
    def on_press(event):
        if event.inaxes != ax:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        rect_x, rect_y = rect.get_xy()
        rect_w, rect_h = rect.get_width(), rect.get_height()
        
        if (rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h):
            state['dragging'] = True
            state['start_pos'] = (x, y)
            state['initial_rect_pos'] = (rect_x, rect_y)
            state['initial_rect_size'] = (rect_w, rect_h)
            state['resize_mode'] = get_resize_mode(x, y, rect_x, rect_y, rect_w, rect_h)
            
            update_info_text()
            ax.figure.canvas.draw_idle()
    
    def on_motion(event):
        if not state['dragging'] or event.inaxes != ax:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        start_x, start_y = state['start_pos']
        dx = x - start_x
        dy = y - start_y
        
        initial_x, initial_y = state['initial_rect_pos']
        initial_w, initial_h = state['initial_rect_size']
        
        min_width = (xlim[1] - xlim[0]) * 0.05
        min_height = (ylim[1] - ylim[0]) * 0.05
        
        if state['resize_mode']:
            # Resize mode
            new_x, new_y = initial_x, initial_y
            new_w, new_h = initial_w, initial_h
            
            if 'e' in state['resize_mode']:
                new_w = max(initial_w + dx, min_width)
            if 'w' in state['resize_mode']:
                new_w = max(initial_w - dx, min_width)
                new_x = initial_x + initial_w - new_w
            if 'n' in state['resize_mode']:
                new_h = max(initial_h + dy, min_height)
            if 's' in state['resize_mode']:
                new_h = max(initial_h - dy, min_height)
                new_y = initial_y + initial_h - new_h
            
            rect.set_xy((new_x, new_y))
            rect.set_width(new_w)
            rect.set_height(new_h)
        else:
            # Move mode
            new_x = initial_x + dx
            new_y = initial_y + dy
            
            new_x = max(xlim[0], min(new_x, xlim[1] - rect.get_width()))
            new_y = max(ylim[0], min(new_y, ylim[1] - rect.get_height()))
            
            rect.set_xy((new_x, new_y))
        
        update_info_text()
        ax.figure.canvas.draw_idle()
    
    def on_release(event):
        state['dragging'] = False
        state['resize_mode'] = None
        state['start_pos'] = None
        state['initial_rect_pos'] = None
        state['initial_rect_size'] = None
        update_info_text()
        ax.figure.canvas.draw_idle()
    
    result = {'saved': False}
    
    def on_key(event):
        if event.key == 'enter':
            x, y = rect.get_xy()
            w, h = rect.get_width(), rect.get_height()
            
            result['saved'] = True
            result['xlim'] = (x, x + w)
            result['ylim'] = (y, y + h)
            result['bbox'] = [x, x + w, y, y + h]
            
            print("\n" + "="*80)
            print("INSET PARAMETERS")
            print("="*80)
            print(f"xlim: ({x:.2f}, {x+w:.2f})")
            print(f"ylim: ({y:.6f}, {y+h:.6f})")
            print(f"bbox: [{x:.2f}, {x+w:.2f}, {y:.6f}, {y+h:.6f}]")
            print("="*80)
            
            plt.close(ax.figure)
        
        elif event.key == 'escape':
            print("Cancelled")
            plt.close(ax.figure)
    
    # Connect events
    fig = ax.figure
    cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
    cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    plt.show()
    
    # Return result
    if result['saved']:
        return {
            'xlim': result['xlim'],
            'ylim': result['ylim'],
            'bbox': result['bbox']
        }
    else:
        # Return current position even if window closed without Enter
        x, y = rect.get_xy()
        w, h = rect.get_width(), rect.get_height()
        return {
            'xlim': (x, x + w),
            'ylim': (y, y + h),
            'bbox': [x, x + w, y, y + h]
        }