program fort_mcmc

implicit none


integer :: N_chains
integer :: N_links
integer :: burn
integer i
double precision, allocatable :: chains(:,:)
double precision :: mean , sig , gauss
double precision, parameter ::  PI  = 4 * atan (1.0_8)

mean = 0
sig = 1.0
gauss =  norm(mean,sig, PI)
N_chains = 1
N_links = 5000
burn = 10

allocate(chains(N_chains,N_links))

chains(1,1) = norm(mean , sig , PI)



do i=2,N_links
    chains(1,i) = norm(mean , sig , PI) + chains(1,i-1)
end do

open(1, file = 'data1.dat', status = 'new')

do i=1, N_links
    write(1,*) chains(i,1)
end do
 
print *, 'Initializing MCMC run with ', N_chains, 'chains and' , N_links, "links"
print *, "Initial chain value" , chains(1,1)
print *, gauss

contains

double precision Function norm(mu, sigma, PI)
    double precision mu, sigma, U1, U2, PI
    call random_number(U1)
    call random_number(U2)
    norm = sigma * sqrt(-2 * log(U1)) * cos(2 * PI * U2) + mu
    RETURN
    END

end program fort_mcmc