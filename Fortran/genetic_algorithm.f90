program genetic_algorithm

!We begin with some variable declarations
implicit none

integer :: popsize
integer :: i
integer :: AllocateStatus

!Define parameter arrays. Need enough to match number of dimensions to the problem.

real, dimension(:), allocatable :: population, gof , param1 , param2
integer :: Ndim



!Some parameters for our Problem:
Ndim = 2
popsize = 100

allocate(population(popsize) , STAT = AllocateStatus)

if (AllocateStatus /= 0) then
    STOP "*** Not enough memory to allocate population array ***"
end if

do i = 1, 100
    population(i) = 10
end do

print *, 'Hello, World, our population size is: ', population(10)

contains

REAL FUNCTION AVRAGE(X,Y,Z)
     REAL X,Y,Z,SUM
     SUM = X + Y + Z
     AVRAGE = SUM /3.0
     RETURN
end program genetic_algorithm