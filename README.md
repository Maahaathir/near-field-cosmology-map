# Near-Field Cosmology 3D Map

**[Click Here to view the Live Interactive 3D Map!](https://maahaathir.github.io/near-field-cosmology-map/)**

This project is an interactive, browser-based 3D visualization of the Local Volume (out to ~15 Mpc) built with Python and Plotly. It transforms standard heliocentric galaxy coordinates into a Local Group (LG) barycentric reference frame to visualize gravitational structures like the Virgo Cluster, Andromeda (M31), and the Milky Way.

## 🚀 Features
* **Coordinate Transformation:** Calculates Solar Apex vectors to shift coordinates from Heliocentric to LG Barycentric frames.
* **Monte Carlo Error Propagation:** Uses a 200-iteration Monte Carlo sampling method to convert distance modulus ($`\mu`$) uncertainties into physical 3D spatial errors ($`\pm`$ Mpc).
* **Interactive UI:** Features a custom dark-mode interface with a dynamic toggling system to instantly snap between reference frames.
* **Depth Slicing:** Includes an interactive spatial slider to filter the cosmological volume from 3 Mpc (Local Group) to 20 Mpc (Virgo Supercluster).

## 🛠️ Installation & Usage
To run the Python script locally and generate the map:

1. Clone the repository:
   ```bash
   git clone https://github.com/Maahaathir/near-field-cosmology-map.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python main.py
   ```
