import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.widgets import TextBox, Button, RadioButtons, Slider
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec


class BispectrumVisualizer:
    def __init__(self):
        # Physics parameters
        self.k1 = 1.0       # Reference scale for k1
        self.fNL = -25.0    # Non-Gaussianity amplitude (default negative for equilateral)
        self.n_s = 0.96     # Spectral index
        self.A_s = 1.0      # Power spectrum amplitude
        
        # Current state
        self.mu = 0.75      # Initial mu value (0.5 <= mu <= 1)
        self.t = 0.95       # Initial t value (0.5 <= t <= 1)
        self.button_pressed = True  # Track if interactivity is enabled
        self.bispectrum_type = 'equilateral'  # Default bispectrum type
        
        # Setup mu-t space boundaries
        self.mu_min, self.mu_max = 0.5, 1.0
        self.t_min, self.t_max = 0.5, 1.0
        
        # Text element references for cleanup
        self.value_text = None
        self.marker_line = None
        self.colorbar = None  # Store reference to colorbar
        
        # Create the figure and axes
        self.setup_figure()
        
        # Setup the triangle visualization space
        self.setup_triangle_space()
        
        # Setup the heatmap
        self.setup_heatmap()
        
        # Setup the results display area
        self.setup_results_space()
        
        # Add controls (radio buttons, text input, sliders, save button)
        self.setup_controls()
        
        # Connect event handlers
        self.connect_events()
        
        # Initial calculations and drawing
        self.calculate_heatmap()
        self.draw_triangle()
    
    def setup_figure(self):
        """Set up the main figure and axes layout"""
        # plt.rcParams['font.size'] = 10
        # self.fig = plt.figure(figsize=(15, 9))
        
        # # Use GridSpec for more control over layout with side-by-side arrangement
        # gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.5], width_ratios=[1, 1])
        
        # # Create the axes with proper spacing
        # self.ax_heatmap = self.fig.add_subplot(gs[0, 0])     # heatmap on left
        # self.ax_triangle = self.fig.add_subplot(gs[0, 1])    # k-triangle on right
        # self.ax_results = self.fig.add_subplot(gs[1, :])     # results area across bottom
        
        # # Adjust spacing
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.2, hspace=0.3, wspace=0.2)
        
        plt.rcParams['font.size'] = 10
        self.fig = plt.figure(figsize=(15, 12))  # Increased height for better proportions
        
        # Use GridSpec for more control over layout with side-by-side arrangement
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.5], width_ratios=[1, 1])
        
        # Create the axes with proper spacing
        self.ax_heatmap = self.fig.add_subplot(gs[0, 0])     # heatmap on left
        self.ax_triangle = self.fig.add_subplot(gs[0, 1])    # k-triangle on right
        self.ax_results = self.fig.add_subplot(gs[1, :])     # results area across bottom
        
        # Set aspect ratio to 1:1 for the main plots
        self.ax_heatmap.set_aspect('equal')
        self.ax_triangle.set_aspect('equal')
        
        # Adjust spacing
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.2, hspace=0.3, wspace=0.2)
    
    def setup_triangle_space(self):
        """Set up the triangle visualization space"""
        self.ax_triangle.set_xlim([0, 1.1])
        self.ax_triangle.set_ylim([0, 1.0])
        self.ax_triangle.set_title('$(k_1, k_2, k_3)$ Triangle', fontsize=12, pad=10)
        
        # Add reference lines for equilateral triangle
        self.ax_triangle.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--', alpha=0.5, linewidth=1)
        self.ax_triangle.plot([1, 0.5], [0, np.sqrt(3)/2], 'k--', alpha=0.5, linewidth=1)
        
        # Initialize the triangle vertices
        self.k_vertices = np.array([
            (0, 0),  # left point
            (self.k1, 0),  # right point
            (0, 0)  # top point (will be updated)
        ])
        
        # Initialize the triangle patch
        self.triangle = Polygon(
            self.k_vertices,
            closed=True,
            edgecolor='black',
            facecolor='blueviolet',
            alpha=0.5,
            linewidth=1
        )
        self.ax_triangle.add_patch(self.triangle)
    
    def setup_heatmap(self):
        """Set up the heatmap visualization (now includes mu-t space functionality)"""
        self.ax_heatmap.set_title(r'$\mu$-$t$ Space: Normalized Bispectrum Heatmap', fontsize=12, pad=10)
        self.ax_heatmap.set_xlabel(r'$\mu$', fontsize=12)
        self.ax_heatmap.set_ylabel(r'$t$', fontsize=12)
        self.ax_heatmap.set_xlim([self.mu_min, self.mu_max])
        self.ax_heatmap.set_ylim([self.t_min, self.t_max])
        
        # Placeholder for heatmap
        self.heatmap_data = None
        self.heatmap_plot = None
        
        # Add cursor position marker
        self.heatmap_marker, = self.ax_heatmap.plot(self.mu, self.t, 'ro')
    
    def setup_results_space(self):
        """Set up the results display area"""
        self.ax_results.set_title('Bispectrum Results', fontsize=12, pad=10)
        self.ax_results.axis('off')
    
    def setup_controls(self):
        """Set up UI controls"""
        # Radio buttons for bispectrum type (bottom left)
        radio_ax = self.fig.add_axes([0.05, 0.08, 0.15, 0.1])
        self.radio = RadioButtons(radio_ax, ('Local', 'Equilateral', 'Orthogonal'), active=1)
        self.radio.on_clicked(self.on_radio_click)
        
        # Add a border around radio buttons
        radio_ax.patch.set_edgecolor('black')
        radio_ax.patch.set_linewidth(0.5)
        
        # Text box for direct mu, t input (bottom center)
        text_box_ax = self.fig.add_axes([0.4, 0.03, 0.3, 0.05])
        self.text_box = TextBox(text_box_ax, 'Input comma separated $\mu$, $t$ pair:', initial='')
        self.text_box.on_submit(self.on_text_submit)
        
        # Save button (bottom right)
        button_ax = self.fig.add_axes([0.75, 0.03, 0.1, 0.05])
        self.save_button = Button(button_ax, 'Save Plot')
        self.save_button.on_clicked(self.save_plot)
        
        # Add parameter sliders
        self.setup_sliders()
    
    def setup_sliders(self):
        """Set up parameter sliders"""
        slider_width = 0.6
        slider_height = 0.03
        slider_x = 0.3
        
        # Spectral index slider
        ns_ax = self.fig.add_axes([slider_x, 0.14, slider_width, slider_height])
        self.ns_slider = Slider(ns_ax, '$n_s$', 0.9, 1.1, valinit=self.n_s, valfmt='%0.3f')
        self.ns_slider.on_changed(self.update_parameters)
        
        # k1 slider
        k1_ax = self.fig.add_axes([slider_x, 0.10, slider_width, slider_height])
        self.k1_slider = Slider(k1_ax, '$k_1$ [h/Mpc]', 0.1, 5.0, valinit=self.k1, valfmt='%0.2f')
        self.k1_slider.on_changed(self.update_parameters)
        
        # fNL slider
        fnl_ax = self.fig.add_axes([slider_x, 0.06, slider_width, slider_height])
        self.fnl_slider = Slider(fnl_ax, '$f_{NL}$', -100, 100, valinit=self.fNL, valfmt='%0.1f')
        self.fnl_slider.on_changed(self.update_parameters)
        
        # Set slider colors
        self.ns_slider.poly.set_color('lightblue')
        self.k1_slider.poly.set_color('lightblue')
        self.fnl_slider.poly.set_color('lightblue')
        
        # Add value indicators to the right of sliders
        self.ns_value_text = self.fig.text(slider_x + slider_width + 0.02, 0.14 + slider_height/2, 
                                          f"{self.n_s:.3f}", ha='left', va='center')
        self.k1_value_text = self.fig.text(slider_x + slider_width + 0.02, 0.10 + slider_height/2, 
                                          f"{self.k1:.2f}", ha='left', va='center')
        self.fnl_value_text = self.fig.text(slider_x + slider_width + 0.02, 0.06 + slider_height/2, 
                                           f"{self.fNL:.1f}", ha='left', va='center')
    
    def connect_events(self):
        """Connect matplotlib event handlers"""
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    
    def on_hover(self, event):
        """Handle hover events in the heatmap (former mu-t space)"""
        if not self.button_pressed or event.inaxes != self.ax_heatmap:
            return
            
        mu, t = event.xdata, event.ydata
        if self.is_valid_mut(mu, t):
            self.mu, self.t = mu, t
            self.draw_triangle()
    
    def on_click(self, event):
        """Handle click events"""
        if event.inaxes == self.ax_heatmap:
            # Get coordinates
            mu, t = event.xdata, event.ydata
            
            # Toggle interactive mode
            self.button_pressed = not self.button_pressed
            
            if self.button_pressed:
                self.ax_triangle.set_facecolor("white")
            else:
                self.ax_triangle.set_facecolor("thistle")
            
            # Process click if coordinates are valid
            if self.is_valid_mut(mu, t):
                self.mu, self.t = mu, t
                self.draw_triangle()
    
    def on_text_submit(self, text):
        """Handle text input submission"""
        try:
            mu, t = map(float, text.split(','))
            if self.is_valid_mut(mu, t):
                self.mu, self.t = mu, t
                self.button_pressed = False
                self.ax_triangle.set_facecolor("thistle")
                self.draw_triangle()
            else:
                print(f"Invalid mu-t pair: {mu}, {t}")
        except ValueError:
            print("Invalid input. Please enter valid mu, t values as comma-separated floats.")
    
    def on_radio_click(self, label):
        """Handle radio button selection"""
        self.bispectrum_type = label.lower()
        self.calculate_heatmap()
        self.draw_triangle()
    
    def update_parameters(self, val=None):
        """Update parameters from sliders"""
        self.fNL = self.fnl_slider.val
        self.k1 = self.k1_slider.val
        self.n_s = self.ns_slider.val
        
        # Update text displays
        self.ns_value_text.set_text(f"{self.n_s:.3f}")
        self.k1_value_text.set_text(f"{self.k1:.2f}")
        self.fnl_value_text.set_text(f"{self.fNL:.1f}")
        
        self.calculate_heatmap()
        self.draw_triangle()
    
    def save_plot(self, event):
        """Save the visualization as an image"""
        self.fig.savefig('Images/bispectrum_viz.png', dpi=150, bbox_inches='tight')
        print("Plot saved at 'Images/bispectrum_viz.png'")
    
    def is_valid_mut(self, mu, t):
        """Check if a mu-t pair is in the valid region"""
        if mu is None or t is None:
            return False
            
        invalid_conditions = [
            mu*t < 0.5,
            mu < self.mu_min,
            t < self.t_min,
            mu > self.mu_max,
            t > self.t_max,
        ]
        return not any(invalid_conditions)
    
    def calculate_power_spectrum(self, k):
        """Calculate power spectrum P(k)"""
        # return self.A_s * k**(-3) 
        if(k==0): return 0
        k0 = 1      # Pivot scale (arbitrary choice for demonstration)
        return self.A_s * k**(-3) * (k/k0)**(self.n_s-1)
    
    def calculate_bispectrum(self, k1, k2, k3, P1, P2, P3):
        """Calculate bispectrum based on selected type"""
        if self.bispectrum_type == 'local':
            # B_local = 2 * fNL * (P1*P2 + P2*P3 + P3*P1)
            return 2 * self.fNL * (P1*P2 + P2*P3 + P3*P1)
        
        elif self.bispectrum_type == 'equilateral':
            # B_equil = 6*fNL*[-(P1*P2 + P2*P3 + P3*P1) - 2*(P1*P2*P3)^(2/3) + (P1^(1/3)*P2^(2/3)*P3 + cyc.)]
            term1 = -(P1*P2 + P2*P3 + P3*P1)
            term2 = -2 * (P1*P2*P3)**(2/3)
            term3 = (P1**(1/3)*P2**(2/3)*P3 + P2**(1/3)*P3**(2/3)*P1 + P3**(1/3)*P1**(2/3)*P2)
            return 6 * self.fNL * (term1 + term2 + term3)
        
        elif self.bispectrum_type == 'orthogonal':
            # B_ortho = 6*fNL*[-3*(P1*P2 + P2*P3 + P3*P1) - 8*(P1*P2*P3)^(2/3) + 3*(P1^(1/3)*P2^(2/3)*P3 + cyc.)]
            term1 = -3 * (P1*P2 + P2*P3 + P3*P1)
            term2 = -8 * (P1*P2*P3)**(2/3)
            term3 = 3 * (P1**(1/3)*P2**(2/3)*P3 + P2**(1/3)*P3**(2/3)*P1 + P3**(1/3)*P1**(2/3)*P2)
            return 6 * self.fNL * (term1 + term2 + term3)
        
        return 0
    
    def midpoint(self, p1, p2):
        """Calculate the midpoint between two points"""
        return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
    
    def distance(self, p1, p2):
        """Calculate the Euclidean distance between two points"""
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def calculate_heatmap(self):
        """Calculate normalized bispectrum values for all valid mu-t points"""
        # Create a grid of mu-t values
        res = 100  # Resolution of the heatmap
        mu_grid = np.linspace(self.mu_min, self.mu_max, res)
        t_grid = np.linspace(self.t_min, self.t_max, res)
        MU, T = np.meshgrid(mu_grid, t_grid)
        
        # Initialize results array
        res_grid = np.zeros_like(MU)
        res_grid.fill(np.nan)  # Start with NaN for invalid regions
        
        # Calculate for each valid point
        for i in range(MU.shape[0]):
            for j in range(MU.shape[1]):
                mu, t = MU[i,j], T[i,j]
                if mu*t >= 0.5:  # Valid region
                    # Calculate k values
                    k1 = self.k1
                    k2 = t * k1
                    k3 = np.sqrt(k1**2 + k2**2 - 2*k1*k2*mu)
                    
                    # Calculate power spectra
                    P1 = self.calculate_power_spectrum(k1)
                    P2 = self.calculate_power_spectrum(k2)
                    P3 = self.calculate_power_spectrum(k3)
                    
                    # Calculate bispectrum
                    B = self.calculate_bispectrum(k1, k2, k3, P1, P2, P3)
                    
                    # Calculate normalized result
                    normalization = P1*P2 + P2*P3 + P3*P1
                    if normalization != 0:
                        res_grid[i,j] = B / normalization
        
        # Update the heatmap plot
        self.ax_heatmap.clear()
        self.ax_heatmap.set_title(r'$\mu$-$t$ Space: Normalized Bispectrum Heatmap', fontsize=12, pad=10)
        self.ax_heatmap.set_xlabel(r'$\mu$', fontsize=12)
        self.ax_heatmap.set_ylabel(r'$t$', fontsize=12)
        self.ax_heatmap.set_xlim([self.mu_min, self.mu_max])
        self.ax_heatmap.set_ylim([self.t_min, self.t_max])
        
        # First, color the invalid region light red
        mu_vals = np.linspace(self.mu_min, self.mu_max, 500)
        t_vals = np.linspace(self.t_min, self.t_max, 500)
        MU_mask, T_mask = np.meshgrid(mu_vals, t_vals)
        valid_mask = MU_mask * T_mask >= 0.5
        self.ax_heatmap.contourf(MU_mask, T_mask, valid_mask, levels=[-1, 0, 1], 
                                colors=['lightcoral', 'white'], alpha=0.5)
        
        # Get data range for colormap
        valid_data = res_grid[~np.isnan(res_grid)]
        if len(valid_data) > 0:
            # Determine min/max values for better visualization
            vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)
            
            # Adjust scale for better visualization
            data_range = vmax - vmin
            if data_range < 20:  # Small range
                vmin = max(vmin - 0.1 * data_range, 0)
                vmax = vmax + 0.1 * data_range
            
            # Round to nice values for colorbar
            vmin = np.floor(vmin / 5) * 5
            vmax = np.ceil(vmax / 5) * 5
            
            # Create colormap
            norm = Normalize(vmin=vmin, vmax=vmax)
            
            # Plot the heatmap
            self.heatmap_plot = self.ax_heatmap.pcolormesh(
                mu_grid, t_grid, res_grid,
                cmap='viridis', 
                norm=norm,
                shading='auto'
            )
            
            # Add colorbar - COMBINED APPROACH FOR SINGLE COLORBAR
            if self.colorbar is not None:
                self.colorbar.remove()
                
            # Create a single colorbar
            self.colorbar = self.fig.colorbar(
                self.heatmap_plot, 
                ax=self.ax_heatmap,
                orientation='vertical',
                pad=0.01
            )
            self.colorbar.set_label('Normalized Bispectrum', fontsize=10)
            
            # Plot mu*t = 0.5 boundary line
            x = np.linspace(self.mu_min, self.mu_max, 100)
            y = 0.5 / x
            self.ax_heatmap.plot(x, y, 'k-', linewidth=1.5)
            
            # Plot mu = t line
            self.ax_heatmap.plot([self.mu_min, self.mu_max], [self.t_min, self.t_max], 
                               'k--', alpha=0.7, linewidth=1)
            
            # Add explanation text about valid regions
            self.ax_heatmap.text(0.03, 0.97, 
                                r"Valid region: $\mu \cdot t \geq 0.5$", 
                                transform=self.ax_heatmap.transAxes,
                                fontsize=10, ha='left', va='top',
                                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            
            # Add marker for current position
            self.heatmap_marker, = self.ax_heatmap.plot(self.mu, self.t, 'ro')
            
            # Store data for reference - we need this for updating the marker
            self.heatmap_data = (mu_grid, t_grid, res_grid, vmin, vmax)
            
            # We don't need the mini colorbar anymore since we have a single combined one
            
        self.fig.canvas.draw_idle()
    
    def update_colorbar_marker(self, res_value):
        """Update the marker on the colorbar to show the current value"""
        if self.colorbar is None or self.heatmap_data is None:
            return
            
        vmin, vmax = self.heatmap_data[3], self.heatmap_data[4]
        
        # Remove previous marker line if it exists
        if self.marker_line is not None:
            self.marker_line.remove()
            self.marker_line = None
            
        # Remove previous value text if it exists
        if self.value_text is not None:
            self.value_text.remove()
            self.value_text = None
            
        # Add marker line on colorbar for current value if in range
        if vmin <= res_value <= vmax:
            # Calculate normalized position (0-1) of value in range
            value_pos = (res_value - vmin) / (vmax - vmin)
            
            # Get colorbar axis position
            cbar_ax = self.colorbar.ax
            
            # Calculate the position for the arrow marker pointing at the colorbar
            y_pos = cbar_ax.get_position().y0 + value_pos * cbar_ax.get_position().height
            
            # Add an arrow pointing to the value on the colorbar
            from matplotlib.patches import FancyArrowPatch
            self.marker_line = self.fig.add_artist(
                FancyArrowPatch(
                    (cbar_ax.get_position().x0 - 0.03, y_pos),  # start point (left of colorbar)
                    (cbar_ax.get_position().x0, y_pos),         # end point (at colorbar edge)
                    arrowstyle='->',
                    color='red',
                    linewidth=2,
                    transform=self.fig.transFigure
                )
            )
            
            # Add text with the value below the colorbar
            self.value_text = self.fig.text(
                (cbar_ax.get_position().x0 + cbar_ax.get_position().x1) / 2,  # Center horizontally
                cbar_ax.get_position().y0 - 0.02,  # Position below colorbar
                f"{res_value:.4f}",
                color='red',
                fontsize=12,
                weight='bold',
                ha='center',
                va='top',
                transform=self.fig.transFigure
            )
    
    def draw_triangle(self):
        """Update the visualization based on current mu, t values"""
        # Update point in heatmap
        if hasattr(self, 'heatmap_marker'):
            self.heatmap_marker.set_data(self.mu, self.t)
        
        # Clear previous triangle display
        self.ax_triangle.clear()
        self.ax_triangle.set_title('$(k_1, k_2, k_3)$ Triangle', fontsize=12, pad=10)
        self.ax_triangle.set_xlim([0, 1.1])
        self.ax_triangle.set_ylim([0, 1.0])
        
        # Redraw reference lines for equilateral triangle
        self.ax_triangle.plot([0, 0.5], [0, np.sqrt(3)/2], 'k--', alpha=0.5, linewidth=1)
        self.ax_triangle.plot([1, 0.5], [0, np.sqrt(3)/2], 'k--', alpha=0.5, linewidth=1)
        
        # Calculate triangle vertices (normalized to fit in the plot)
        k1_norm = 1.0  # Always show k1 = 1.0 in the display
        
        self.k_vertices = np.array([
            (0, 0),  # left (origin)
            (k1_norm, 0),  # right (along x-axis)
            (self.t*self.mu*k1_norm, self.t*k1_norm*np.sqrt(1-self.mu**2))  # top
        ])
        
        # Create and add the triangle patch
        triangle = Polygon(
            self.k_vertices,
            closed=True,
            edgecolor='black',
            facecolor='blueviolet',
            alpha=0.5,
            linewidth=1
        )
        self.ax_triangle.add_patch(triangle)
        
        # Calculate actual k values
        k1 = self.k1
        k2 = self.t * k1
        k3 = np.sqrt(k1**2 + k2**2 - 2*k1*k2*self.mu)
        
        # Label triangle sides
        self.ax_triangle.text(*self.midpoint(self.k_vertices[0], self.k_vertices[1]), 
                              r'$k_1$', ha='center', va='bottom', fontsize=12)
        self.ax_triangle.text(*self.midpoint(self.k_vertices[0], self.k_vertices[2]), 
                              r'$k_2$', ha='left', va='top', fontsize=12)
        self.ax_triangle.text(*self.midpoint(self.k_vertices[1], self.k_vertices[2]), 
                              r'$k_3$', ha='right', va='top', fontsize=12)
        
        # Calculate power spectra
        P1 = self.calculate_power_spectrum(k1)
        P2 = self.calculate_power_spectrum(k2)
        P3 = self.calculate_power_spectrum(k3)
        
        # Calculate bispectrum
        B = self.calculate_bispectrum(k1, k2, k3, P1, P2, P3)
        
        # Calculate normalized result (res)
        normalization = P1*P2 + P2*P3 + P3*P1
        res = B / normalization if normalization != 0 else 0
        
        # Display mu-t and k values in triangle plot
        self.ax_triangle.text(0.05, 0.95, 
                              fr"$\mu$:  {self.mu:.6f}" + f"\n$t$:  {self.t:.6f}", 
                              ha='left', va='top', fontsize=12, family='monospace',
                              transform=self.ax_triangle.transAxes)
        
        self.ax_triangle.text(0.95, 0.95, 
                              fr"$k_1$:  {k1:.6f}" + f"\n$k_2$:  {k2:.6f}\n$k_3$:  {k3:.6f}", 
                              ha='right', va='top', fontsize=12, family='monospace',
                              transform=self.ax_triangle.transAxes)
        
        # Update the results display
        self.ax_results.clear()
        self.ax_results.axis('off')
        
        # Update the colorbar marker with the current value
        self.update_colorbar_marker(res)
        
        # Display results text with clear formatting
        # Left side information
        result_text_1 = (
            f"Bispectrum Type: {self.bispectrum_type.capitalize()}"
        )
        
        param_text = (
            f"Parameters:\n"
            f"  $f_{{NL}}$ = {self.fNL:.2f}\n"
            f"  $k_1$ = {self.k1:.2f} h/Mpc\n"
            f"  $n_s$ = {self.n_s:.3f}"
        )
        
        # Right side information
        config_text = (
            f"Triangle Configuration:\n"
            f"  $\mu$ = {self.mu:.6f}\n"
            f"  $t$ = {self.t:.6f}"
        )
        
        power_text = (
            f"Power Spectra:\n"
            f"  P($k_1$) = {P1:.4e}\n"
            f"  P($k_2$) = {P2:.4e}\n"
            f"  P($k_3$) = {P3:.4e}"
        )
        
        # Place texts in good positions
        self.ax_results.text(0.05, 0.75, param_text,
                           transform=self.ax_results.transAxes,
                           fontsize=11, family='monospace', va='top')
        
        self.ax_results.text(0.35, 0.75, config_text,
                           transform=self.ax_results.transAxes,
                           fontsize=11, family='monospace', va='top')
        
        self.ax_results.text(0.60, 0.75, power_text,
                           transform=self.ax_results.transAxes,
                           fontsize=11, family='monospace', va='top')
        
        # Add usage instructions to results area
        instruction_text = (
            "Instructions: Click on heatmap to select a point; click again to toggle interactive mode."
            "\nEnter comma-separated values in the text box below for precise input."
        )
        self.ax_results.text(0.5, 0.15, instruction_text,
                            transform=self.ax_results.transAxes,
                            fontsize=10, ha='center', va='top',
                            bbox=dict(facecolor='lightyellow', alpha=0.7, boxstyle='round'))
        
        # Draw the figure
        self.fig.canvas.draw_idle()


# Run the visualization
if __name__ == "__main__":
    visualizer = BispectrumVisualizer()
    plt.show()