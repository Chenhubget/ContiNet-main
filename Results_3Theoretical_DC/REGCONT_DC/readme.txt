computer code, supporting files and datasets associated
with the manuscript:

"REGCONT: A MATLAB based program for stable downward continuation of geophysical potential fields using Tikhonov regularization"

authors:  Pasteka R., Karcol R., Kusnirak D., Mojzes A.,
Department of applied and environmental geophysics, Comenius University, Mlynská dol., 842 15 Bratislava, Slovak Republic
and Department Geophysical Institute, Slovak Academy of Sciences, Dúbravská cesta 9, 845 28 Bratislava, Slovak Republic

corresponding author: Pasteka R.,	e-mail: pasteka@fns.uniba.sk

1. main MATLAB script:  REGCONT.m
2. suuporting files:  main_logo.rlg   (logo in the GUI environment; should be in the same working directory),
                      REGCONT1_0.clr  (color file for grid vizulaization in Matlab images - file format is identical with GS Surfer format)
3. datasets:
   a) 'fig1_and 2_input.dat'   - input ASCII data file with the example in figures 1 and 2,
      'fig1_output_all_calculated_data.dat'  - all evaluated data used in Fig. 1 (synthetic and downward continued fields without regularization),
      'fig2_output_all_calculated_data.dat'  - all evaluated data used in Fig. 2 (synthetic and downward continued fields by means of the regularization approach),
      'fig3_output_all_Cnorms.dat'     -  all evaluated C-norms, connected with the Fig. 2 and 3,
   b) 'fig7_input.grd'    - input ASCII GS Surfer grid file with the example in Fig. 7 (used also for the creation of Fig. 5 and 6),
      'fig8_output_all_Cnorms.dat'     -  all evaluated C-norms, connected with the Fig. 7 and 8,
   c) 'fig9_input_niv0_7m_measured.grd'  - measured dT magnetic field from the Fig.9, level 0.7 m (ASCII GS Surfer grid file),
      'fig9_input_niv1_0m_measured.grd'  - measured dT magnetic field from the Fig.9, level 1.0 m (ASCII GS Surfer grid file),
      'fig9_output_reg_cont_downto_from1_0_to_0_7m.grd'  - downward continued T magnetic field from the Fig.9, from level 1.0 m downto 0.7 m (ASCII GS Surfer grid file),
      'fig10_displayed_data.dat'   -  interpolated data (measured and downward continued) from the result in Fig. 9 along the profile x = 105 m,
      'fig11_output_all_C_and_L1norms.dat'   - all evaluated C-norms and L1-norms, connected with the Fig. 9,

