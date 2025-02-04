program fort_rejection

implicit none

integer :: Nsamp
integer :: Ndim
integer :: i
integer :: Ntrial
double precision :: a,b,a1,b1
double precision :: fx , samp
double precision :: mu , sig
double precision, allocatable :: sample(:,:)


Nsamp = 100000
Ndim = 1
a = -50
b = 50

a1 = 0
b1 = 0.10
Ntrial = 0
mu = -3
sig = 10.0

allocate(sample(Ndim,Nsamp))

print *, "Generating " , Nsamp ,  "samples across " , Ndim , "Dimensions"


i = 1
do while (sample(1,Nsamp) == 0)
    sample(1,i) = uniform(a , b)
    
    samp = uniform(a1 , b1)
    fx = gauss(mu , sig , sample(1,i))

    if (samp < fx) then
        i = i + 1
    end if

    Ntrial = Ntrial + 1
end do

print *, Ntrial

! Output Results to a file

open(1, file = 'rsamp.dat', status = 'new')

do i=1, Nsamp
    write(1,*) sample(1,i)
end do

contains


double precision Function uniform(a , b)
    double precision :: a , b

    call random_number(uniform)
    uniform = uniform * (b - a)
    uniform = uniform +  a
    RETURN
    END

double precision Function gauss(mu, sigma, x)
    double precision :: mu, sigma, x

    gauss = 1 / (sqrt(8 * atan (1.0_8) * ( sigma ** 2 )))
    gauss = gauss * exp(-1 * ( (  x - mu ) ** 2 / (2 * sigma ** 2) ) )

    RETURN
    END

end program fort_rejection