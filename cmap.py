import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D  # Para gráficos 3D

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Puntos dispersos (x, y) y sus valores z
x = np.random.rand(100) * 10  # Coordenadas x en un rango de 0 a 10
y = np.random.rand(100) * 10  # Coordenadas y en un rango de 0 a 10
z = np.sin(x) + np.cos(y)     # Valores de la función (puede ser cualquier dato)

# Crear una cuadrícula
xi = np.linspace(min(x), max(x), 100)
yi = np.linspace(min(y), max(y), 100)
Xi, Yi = np.meshgrid(xi, yi)

# Interpolación de los valores en la cuadrícula
Zi = griddata((x, y), z, (Xi, Yi), method='cubic')  # Métodos: 'nearest', 'linear', 'cubic'

# Graficar el mapa de calor
plt.figure(figsize=(8,6))
plt.contourf(Xi, Yi, Zi, levels=50, cmap='jet')  # Usa 'pcolormesh' si prefieres un mapa sin interpolación suave
plt.colorbar(label="Valor de Z")
plt.scatter(x, y, c='black', s=10, label="Puntos originales")  # Muestra los puntos originales
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Mapa de calor a partir de puntos dispersos")
plt.show()
'''
# Coordenadas polares para sacar la ubicacion de los puntos en el plano x,y
x2, x3, x4, x5, x6, x1 = [np.cos(th) for th in np.linspace(0, 2*np.pi, 7)][0: -1] #2, 3, 4, 5, 6, 1
y2, y3, y4, y5, y6, y1 = [np.sin(th) for th in np.linspace(0, 2*np.pi, 7)][0: -1]

x = [np.cos(th) for th in np.linspace(0, 2*np.pi, 7)][0: -1] #2, 3, 4, 5, 6, 1
y = [np.sin(th) for th in np.linspace(0, 2*np.pi, 7)][0: -1]

x12 = (x1 + x2)/2  #x15 = x51 coordenadas iguales. Lo mismo para y15 y y51
y12 = (y1 + y2)/2

x13 = (x1 + x3)/2 
y13 = (y1 + y3)/2

x14 = (x1 + x4)/2 
y14 = (y1 + y4)/2

x15 = (x1 + x5)/2 
y15 = (y1 + y5)/2

x16 = (x1 + x6)/2 
y16 = (y1 + y6)/2

x23 = (x2 + x3)/2 
y23 = (y2 + y3)/2

x24 = (x2 + x4)/2 
y24 = (y2 + y4)/2

x25 = (x2 + x5)/2 
y25 = (y2 + y5)/2

x26 = (x2 + x6)/2 
y26 = (y2 + y6)/2

x34 = (x3 + x4)/2 
y34 = (y3 + y4)/2

x35 = (x3 + x5)/2 
y35 = (y3 + y5)/2

x36 = (x3 + x6)/2 
y36 = (y3 + y6)/2

x45 = (x5 + x4)/2 
y45 = (y5 + y4)/2

x46 = (x6 + x4)/2 
y46 = (y6 + y4)/2

x56 = (x5 + x6)/2 
y56 = (y5 + y6)/2


xper = x.copy()
yper = y.copy()
xper.append(x[0])
yper.append(y[0])

x = [x12, x13, x14, x15, x16, x23, x24, x26, x34, x35, x45, x46, x56] #14, 25, y 36 son iguales
y = [y12, y13, y14, y15, y16, y23, y24, y26, y34, y35, y45, y46, y56]

plt.plot(xper, yper,color='green')
plt.scatter(x, y, c='magenta')
plt.show()

#df_filtered = pd.DataFrame()

df = pd.read_csv('prueba_valen.csv', sep=';',index_col=False)
df_filtered = df[(df['freq'] == 5000) & (df['lado'] == 'der')]

# electrode_pairs = {
#     12: 21, 13: 31, 14: 41, 15: 51, 16: 61,
#     23: 32, 24: 42, 25: 52, 26: 62,
#     34: 43, 35: 53, 36: 63,
#     45: 54, 46: 64,
#     56: 65
# }

electrode_pairs = {
    12: 21, 13: 31, 14: 41, 15: 51, 16: 61,
    23: 32, 24: 42, 26: 62,
    34: 43, 35: 53,
    45: 54, 46: 64,
    56: 65
}

print(electrode_pairs.items())
# df_filtered['electro'].values
# print(df.head())

zd = []

for e1, e2 in electrode_pairs.items():
    if e1 == 14 and e2 == 41:
        imps = {14:41,25:52,36:63}
        imps_values =[]
        for e11, e21 in imps.items():
            if (df_filtered['electro'] == e11).any() and (df_filtered['electro'] == e21).any():
                    imp = (df_filtered.loc[df_filtered['electro'] == e11, 'imp'].iloc[0] +
                        df_filtered.loc[df_filtered['electro'] == e21, 'imp'].iloc[0]) / 2
                    imps_values.append(imp)
        print(imps_values)
        avg_imp14 = sum(imps_values)/3
        print(avg_imp14)
        zd.append(avg_imp14)
    else:    
        if e1 in df_filtered['electro'].values and e2 in df_filtered['electro'].values:
            imp1 = df_filtered[df_filtered['electro'] == e1]['imp'].values[0]
            imp2 = df_filtered[df_filtered['electro'] == e2]['imp'].values[0]
            avg_imp = (imp1 + imp2) / 2
            zd.append(avg_imp)
print(zd)

dfi = pd.read_csv('prueba_valen.csv', sep=';',index_col=False)
dfi_filtered = dfi[(dfi['freq'] == 5000) & (dfi['lado'] == 'izq')]

zi = []
for e1, e2 in electrode_pairs.items():
    if e1 == 14 and e2 == 41:
        imps = {14:41,25:52,36:63}
        imps_values =[]
        for e11, e21 in imps.items():
            if (dfi_filtered['electro'] == e11).any() and (dfi_filtered['electro'] == e21).any():
                    imp = (dfi_filtered.loc[dfi_filtered['electro'] == e11, 'imp'].iloc[0] +
                        dfi_filtered.loc[dfi_filtered['electro'] == e21, 'imp'].iloc[0]) / 2
                    imps_values.append(imp)
        print(imps_values)
        avg_imp14 = sum(imps_values)/3
        print(avg_imp14)
        zi.append(avg_imp14)
    else:    
        if e1 in dfi_filtered['electro'].values and e2 in dfi_filtered['electro'].values:
            imp1 = dfi_filtered[dfi_filtered['electro'] == e1]['imp'].values[0]
            imp2 = dfi_filtered[dfi_filtered['electro'] == e2]['imp'].values[0]
            avg_imp = (imp1 + imp2) / 2
            zi.append(avg_imp)


(z12, z13, z14, z15, z16, 
 z23, z24, z26, z34, z35, 
 z45, z46, z56) = zd

(z12, z13, z14, z15, z16, 
 z23, z24, z26, z34, z35, 
 z45, z46, z56) = zi

print(zd)
print(zi)

# Crear la cuadricula
xi = np.linspace(-1, 1, 100)
yi = np.linspace(-1, 1, 100)
Xi, Yi = np.meshgrid(xi, yi)

# Interpolacion de los valores en la cuadricula
Zid = griddata((x, y), zd, (Xi, Yi), method='linear')  # Metodos: 'nearest', 'linear', 'cubic'
Zi = griddata((x, y), zi, (Xi, Yi), method='linear')

# # Graficar el mapa de calor
# plt.figure(figsize=(8,6))
# plt.contourf(Xi, Yi, Zid, levels=50, cmap='plasma')  # 'pcolormesh' para un mapa sin interpolación suave #jet
# plt.colorbar(label="Valor de impedancia (altura)")
# plt.scatter(x, y, c='black', s=10, label="Puntos promedios del barrido")  # Muestra los puntos originales
# plt.legend()
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Mapa de calor derecho a partir de puntos promedios")
# plt.show()

# Ambos senos
plt.figure(figsize=(12, 5))
#zd
plt.subplot(1, 2, 1)  
plt.contourf(Xi, Yi, Zid, levels=50, cmap='inferno')
plt.colorbar(label="Valor de impedancia (Zd)")
plt.scatter(x, y, c='black', s=10, label="Puntos promedios")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Mapa de calor - Zd")

#zi
plt.subplot(1, 2, 2) 
plt.contourf(Xi, Yi, Zi, levels=50, cmap='inferno')
plt.colorbar(label="Valor de impedancia (Zi)")
plt.scatter(x, y, c='black', s=10, label="Puntos promedios")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Mapa de calor - Zi")

plt.tight_layout()
plt.show()


#Graficar el mapa de calor en 3D
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Xi, Yi, Zi, cmap='plasma')
ax.scatter(x, y, z, color='black', s=20, label="Puntos promedios del barrido")

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Valor de impedancia")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Impedancia")
ax.set_title("Mapa de calor 3D de impedancia")

ax.view_init(elev=30, azim=220)  

plt.legend()
plt.show()