import matplotlib.pyplot as plt


fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(121)

print type(ax)

# ax.set_xticks(rotation=45)
ax.plot(Prin_Balances['UPB'], '--r', label='UPB')
ax.legend()
ax.tick_params('Bal', colors='r')


# Get second axis
ax2 = fig.add_subplot(122)
ax2.plot(Prin_Balances['1 Mos'],  label='1 Mos', color = 'blue')
ax2.plot(Prin_Balances['2 Mos'],  label='2 Mos', color = 'green')
ax2.plot(Prin_Balances['3 Mos'],  label='3 Mos', color = 'yellow')
ax2.plot(Prin_Balances['> 3 Mos'],  label='>3 Mos', color = 'purple')
plt.legend()

ax.tick_params('vals', colors='b')

plt.show()