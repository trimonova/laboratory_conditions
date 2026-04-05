def rotate(A,B,C):
  return (B[0]-A[0])*(C[1]-B[1])-(B[1]-A[1])*(C[0]-B[0])

def grahamscan(A):
  n = len(A) # число точек
  P = list(range(n))# список номеров точек
  print(P)
  for i in range(1,n):
    if A[P[i]][0]<A[P[0]][0]: # если P[i]-ая точка лежит левее P[0]-ой точки
      P[i], P[0] = P[0], P[i] # меняем местами номера этих точек
  for i in range(2,n): # сортировка вставкой
    j = i
    while j>1 and (rotate(A[P[0]],A[P[j-1]],A[P[j]])<0):
      P[j], P[j-1] = P[j-1], P[j]
      j -= 1
  S = [P[0], P[1]]  # создаем стек
  for i in range(2, n):
      while rotate(A[S[-2]], A[S[-1]], A[P[i]]) < 0:
          del S[-1]  # pop(S)
      S.append(P[i])  # push(S,P[i])
  A_new = []
  for i in P:
      A_new.append(A[i])
  return A_new

def jarvismarch(A):
  n = len(A)
  P = list(range(n))
  # start point
  for i in range(1,n):
    if A[P[i]][0]<A[P[0]][0]:
      P[i], P[0] = P[0], P[i]
  H = [P[0]]
  del P[0]
  P.append(H[0])
  while True:
    right = 0
    for i in range(1,len(P)):
      if rotate(A[H[-1]],A[P[right]],A[P[i]])<0:
        right = i
    if P[right]==H[0]:
      break
    else:
      H.append(P[right])
      del P[right]

  A_new = []
  for i in H:
      A_new.append(A[i])
  return A_new

