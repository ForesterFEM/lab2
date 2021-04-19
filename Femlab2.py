import numpy as np
import matplotlib.pyplot as plt
import zad15 
# # arr = np.array([1,2,3,4,5])
# # print (arr)
# # A =np.array([[1,2,3],[7,8,9]])
# # print(A)
# A=np.array([[1,2,3],
#             [7,8,9]])
# # print(A)
# # A =np.array([[1,2, \# po backslashu'u nie moze byc zadnego znaku!
# #             [7,8,9]])
# # print(A)

# # v =np.arange(1,7)
# # print(v,"\n")
# # v =np.arange(-2,7)
# # print(v,"\n")
# # v =np.arange(1,10,3)
# # print(v,"\n")
# # v =np.arange(1,10.1,3)
# # print(v,"\n")
# # v =np.arange(1,11,3)
# # print(v,"\n")
# # v =np.arange(1,2,0.1)
# # print(v,"\n")

# # v =np.linspace(1,3,4)
# # print(v)
# # v =np.linspace(1,10,4)
# # print(v)

# X= np.ones((2,3))
# Y= np.zeros((2,3,4))
# Z= np.eye(2) 
# R= np.eye(2,3)
# Q= np.random.rand(2,5) 
# # print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q,"\n\n",R)

# # U = np.block([[A], [X,Z]])
# # print(U)

# V =np.block([[
# np.block([
# np.block([[np.linspace(1,3,3)],
# [np.zeros((2,3))]]) ,
# np.ones((3,1))])
# ],
# [np.array([100, 3, 1/2, 0.333])]] )
# print(V)

# # print (V[0,2])
# # print (V[3,0])
# # print (V[3,3])
# # print (V[-1, -1])
# # print (V[-4, -3])

# # print (V[3,:])
# # print (V[:,2])

# # print (V[3,0:3])
# # print (V[np.ix_([0,2,3],[0,-1])])
# # print (V[3])

# # Q= np.delete(V, 2, 0)
# # print(Q)

# # Q= np.delete(V, 2, 1)
# # print(Q)

# v= np.arange(1,7)
# # print( np.delete(v, 3, 0))

# # print( np.size(v))
# # print( np.shape(v))

# # print( np.size(V))
# # print (np.shape(V))


# A = np.array([[1, 0, 0],
# [2, 3, -1],
# [0, 7, 2]] )
# B = np.array([[1, 2, 3],
# [-1, 5, 2],
# [2, 2, 2]] )
# # print( A+B )
# # print( A-B )
# # print( A+2 )
# # print( 2*A )

# # MM1 = A@B
# # print(MM1)
# # MM2 = B@A
# # print(MM2)

# # MT1 =A*B
# # print(MT1)
# # MT2 =B*A
# # print(MT2) 

# # DT1 = A/B
# # print(DT1)

# # C= np.linalg.solve(A,MM1)
# # print(C)

# # x= np. ones((3,1))
# # b= A@x
# # y= np.linalg.solve(A,b)
# # print(y)

# # PM = np.linalg.matrix_power(A, 2)
# # PT=A**2

# # A.T
# # A.transpose()
# # A.conj().T
# # A.conj().transpose()


# # np.logical_not(A)
# # np.logical_and(A, B)
# # np.logical_or(A, B)
# # np.logical_xor(A, B)
# # print( np.all(A) )
# # print( np.any(A) )

# # print(v > 4)
# # print(np.logical_or(v>4, v<2))
# # print(np.nonzero(v>4))
# # print(v[np.nonzero(v>4)])

# print(np.max(A))
# print(np.min(A))
# print(np.max(A,0))
# print(np.max(A,1))
# print(A.flatten())
# print(A.flatten('F'))

# x = [1,2,3]
# y = [4,6,5]
# plt.plot(x,y)
# plt.show()

# x= np.arange(0.0, 2.0, 0.01)
# y1= np.sin(2.0*np.pi*x)
# y2= np.cos(2.0*np.pi*x) 
# plt.plot(x,y1,'r:',x,y2, 'g', linewidth=2)
# plt.legend(('dane y1', 'dane y2'))
# plt.xlabel('czas')
# plt.ylabel('pozycja')
# plt.title('wykres')
# plt.grid(True)
# plt.show()

# x= np.arange(0.0, 2.0, 0.01)
# y1= np.sin(2.0*np.pi*x)
# y2= np.cos(2.0*np.pi*x) 
# y= y1*y2
# l1= plt.plot(x,y,'b')
# l2,l3= plt.plot(x,y1,'r:',x,y2,'g')
# plt.legend((l2,l3,l1),('dane y1','dane y2', 'y1*y2'))
# plt.xlabel('czas')
# plt.ylabel('pozycja')
# plt.title('wykres')
# plt.grid(True)
# plt.show()

#######################################################################################################


# #zadanie 3
# x1 = np.arange(1,6)
# x2 = np.linspace(5,1,5)
# x3 = np.zeros((2,2))
# x4 = np.ones((2,3))*2
# x5 = np.zeros(2)
# x6 = np.linspace(-90,-70,3)
# x7 = np.ones((5,1))*10
# A = np.block([ np.block([[ np.block([[x1],[x2]])],
#               [np.block( [x3,x4])],
#               [np.block([x5,x6])]]), x7])
# print(A)

# #zadanie 4
# B=A[1,:]+A[3,:]
# print(B)

# #zadanie 5
# C = np.max(A,0)
# print(C)

# #zadanie 6
# D = np.delete(B,[0,5],0) 
# print(D)

# #zadanie 7
# for i in range(len(D)):
#     if D[i] == 4:
#        D[i]=0 
# print(D)

# #zadanie 8
# E= np.delete(C,[np.where(C==np.max(C)),np.where (C==np.amin(C))])
# print(E)
            
# #zadanie 9
# test= np.where(A == np.max(A))[0]
# for test2 in np.where(A == np.min(A))[0]:
#     if test2 in test:
#         print(A[test2])


# #zadanie  10
# MT1=D*E
# print(MT1)
# MM1=D@E
# print(MM1)
# MM2=E@D
# print(MM2)

#zadanie 11
# def funkcja(n):
#     x=np.random.randint(10, size=(n,n))
#     return x, sum(x.diagonal())

# #zadanie 12
# def funkcja2(n):
#     np.fill_diagonal(n, 0)
#     np.fill_diagonal(np.fliplr(n),0)
#     return n
    
# #zadanie 13
# def funkcja3(n):
#     a=0    
#     lwier, lkol= n.shape
#     for i in range (0,lwier):
#         print( i)
#         if (i+1)%2==0:
#             b= sum(n[i,:])
#             a=a+b
#     return a

#zadanie 14
x = np.arange(-10,10.1,0.1)
y1 = lambda x: np.cos(2*x) 
# print(x)
# plt.plot(x,y1(x),'r--', linewidth=2)

# #zadanie 15
# plt.plot(x,y1(x),'r--',x,zad15.funkcja(x),'g+', linewidth=2)

# #zadanie 16
o = np.append((lambda x: np.sin(x)) (x[np.where(x < 0)[0]]),(lambda x: np.sqrt(x)) (x[np.where(x >= 0)[0]]))
# plt.plot(x,o)

#zadanie 17
y2 = zad15.funkcja(x)
y3 =lambda x: 3*y1(x)+y2
plt.plot(x,y3(x),'b*')

#zadanie 18
W=np.array([[10, 5, 1, 7],
          [10, 9, 5, 5],    
    [1, 6, 7, 3],
    [10, 0, 1, 5]])
R= [[34],
    [44],
    [25],
    [27]]
xy = np.linalg.solve(W, R)
print(xy)

#haslo to BABA 

#zadanie 19
from math import sin, pi
from scipy.integrate import quad
wynik, blad = quad(sin, 0, 2*pi)
print('Wynik calkowania ', np.round(wynik,5),'\n Blad calkowania', np.round(blad,5))
