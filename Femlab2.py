import numpy as np
# arr = np.array([1,2,3,4,5])
# print (arr)
# A =np.array([[1,2,3],[7,8,9]])
# print(A)
A=np.array([[1,2,3],
            [7,8,9]])
# print(A)
# A =np.array([[1,2, \# po backslashu'u nie moze byc zadnego znaku!
#             [7,8,9]])
# print(A)

# v =np.arange(1,7)
# print(v,"\n")
# v =np.arange(-2,7)
# print(v,"\n")
# v =np.arange(1,10,3)
# print(v,"\n")
# v =np.arange(1,10.1,3)
# print(v,"\n")
# v =np.arange(1,11,3)
# print(v,"\n")
# v =np.arange(1,2,0.1)
# print(v,"\n")

# v =np.linspace(1,3,4)
# print(v)
# v =np.linspace(1,10,4)
# print(v)

X= np.ones((2,3))
Y= np.zeros((2,3,4))
Z= np.eye(2) 
R= np.eye(2,3)
Q= np.random.rand(2,5) 
# print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q,"\n\n",R)

# U = np.block([[A], [X,Z]])
# print(U)

V =np.block([[
np.block([
np.block([[np.linspace(1,3,3)],
[np.zeros((2,3))]]) ,
np.ones((3,1))])
],
[np.array([100, 3, 1/2, 0.333])]] )
print(V)

# print (V[0,2])
# print (V[3,0])
# print (V[3,3])
# print (V[-1, -1])
# print (V[-4, -3])

# print (V[3,:])
# print (V[:,2])

# print (V[3,0:3])
# print (V[np.ix_([0,2,3],[0,-1])])
# print (V[3])

# Q= np.delete(V, 2, 0)
# print(Q)

# Q= np.delete(V, 2, 1)
# print(Q)

# v= np.arange(1,7)
# print( np.delete(v, 3, 0))

# print( np.size(v))
# print( np.shape(v))

# print( np.size(V))
# print (np.shape(V))


A = np.array([[1, 0, 0],
[2, 3, -1],
[0, 7, 2]] )
B = np.array([[1, 2, 3],
[-1, 5, 2],
[2, 2, 2]] )
# print( A+B )
# print( A-B )
# print( A+2 )
# print( 2*A )

MM1 = A@B
print(MM1)
MM2 = B@A
print(MM2)

