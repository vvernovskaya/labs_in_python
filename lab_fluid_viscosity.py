import lab_functions as lf

ln_vis = lf.extract_data_into_array('лаба9.xlsx', 'Лист1', 14)
t_1 = lf.extract_data_into_array('лаба9.xlsx', 'Лист1', 15)

w = lf.calc_b(t_1, ln_vis)
w = w*1.38*(10**(-23))
sigma_w = lf.calc_b_error(t_1, ln_vis)

print(ln_vis)
print(t_1)
print(w)
print(sigma_w)

lf.make_diagram(t_1, ln_vis, 'Зависимость ln(viscosity) от 1/T', '1/T, 1/C',
                'ln(viscosity)')
