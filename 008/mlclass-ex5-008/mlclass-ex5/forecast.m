function [OutPrice,pPR,PeriodAll] = forcast(d,num_years)

d(d==0)=NaN;
Period = datestr(busdays(datenum('1990Q1','yyyyqq'),datenum('2023Q4','yyyyqq'),'Q'),'yyyyqq');
PeriodAll = ...
    datestr(busdays(datenum('1990Q1','yyyyqq'),addtodate(datenum('2013Q1','yyyyqq'),num_years,'y'),'Q'),'yyyyqq');


ptiles = [0.2 1 2 15 50 75 85 90 98 99.8];
l_prct = length(ptiles);
idxT0  = find(strcmp(Period,{'2012Q3'}));

% Set the forecast dates and number
l_fore = num_years*4;
growth_factor_price  = zeros(l_prct,l_fore+1);
q1w = linspace(0.5,1,4);
Price = d(1:idxT0);	% get the hist prices

%Price Return growth and reversion callibration
PriceRet = diff(Price)./Price(1:end-1);
PriceRet = [NaN; PriceRet];
pPR      = CIR_calibration(PriceRet,1);

% find percentile loss with normal distribution, 
% the alternative way is to use historical price chg to find prctile loss (empirical): 
% ie: gam_price = prctile(PriceRet)
[mu,sig] = normfit(PriceRet(~isnan(PriceRet)));
gam_price= norminv(ptiles/100,mu,sig);

OutPrice = zeros(length(Price)+l_fore,l_prct);

for pp=1:l_prct
    
    gamma(1:4) = q1w*gam_price(pp);		% short term
    for jj=5:l_fore
        gamma(jj)=pPR(1)*pPR(2) + (1-pPR(1))*gamma(jj-1);	% long term mean reversion
    end
    
    growth_factor_price(pp,1:l_fore) = cumprod(1+gamma);

    OutPrice(:,pp)=[Price(1:idxT0); ...
        [Price(idxT0)*squeeze(growth_factor_price(pp,1:l_fore))]'];  % combine hist with forecast
    OutPrice(:,pp) =  OutPrice(:,pp)/Price(idxT0); 	% normalize to current price
    
end



function ML_CIRparams = CIR_calibration( V_data ,dt,params)
V_data = V_data(~isnan(V_data));
N = length(V_data);
if nargin <3
    x = [ ones(N-1,1) V_data(1:N-1)];
    ols = (x'*x)^( -1)*(x'*V_data (2:N));
    m =mean(V_data); 
    v= var(V_data);
    params = [- log(ols(2))/dt,m,sqrt(2*ols(2)*v/m)];
end
options = optimset ('MaxFunEvals', 100000 , 'MaxIter', 100,'display','off');

ML_CIRparams = fminsearch( @FT_CIR_LL_ExactFull , params , options );

    function mll = FT_CIR_LL_ExactFull(params)
        alpha = params(1); 
        theta = params(2); 
        sigma = params(3);
        c = (2* alpha )/(( sigma^2)*(1 - exp(-alpha*dt )));
        q = ((2*alpha*theta )/(sigma^2))-1;
        u = c*exp(-alpha*dt)*V_data(1:N -1);
        v = c*V_data(2:N);
        
        %mle function to minimize
        mll = -(N-1)*log(c)+sum(u+v-log(v./u)*q /2 -...
            log( besseli(q,2*sqrt(u.*v),1)) - abs(real(2*sqrt(u.*v))));
    end
end
