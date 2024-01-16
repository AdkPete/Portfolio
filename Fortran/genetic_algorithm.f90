program genetic_algorithm

!We begin with some variable declarations
implicit none

integer :: popsize
integer :: i
integer :: AllocateStatus
real, dimension(:), allocatable :: population

popsize = 100
allocate(population(popsize) , STAT = AllocateStatus)

if (AllocateStatus /= 0) then
    STOP "*** Not enough memory to allocate population array ***"
end if

do i = 1, 100
    population(i) = 10
end do

print *, 'Hello, World, our population size is: ', population(10)

end program genetic_algorithm