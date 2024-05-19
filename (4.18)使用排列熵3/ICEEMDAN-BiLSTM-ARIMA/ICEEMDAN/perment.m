function [H] = perment( data,n )
    m = 6; %embeding
    tau = 1; %delay time
    f = (n-(m-1)*tau);
    x = zeros (f,m);
    x_sort = zeros (f,m);

    for i = 1:f
        for j = 1:m
            x(i,j) = data(i+(j-1)*tau);
        end
    end

    [x_sort, indice] = sort (x,2);
    p = perms(1:m);
    z = zeros(length(p),1);
    for i=1:f
        for j=1:length(p)
          if indice(i,:) == p(j,:)
                z(j) = z(j) + 1;
                break;
          end
        end
    end

    sz = sum(z);
    zpsz = z / sz;
    H = 0;
    for i = 1:length(zpsz)
      if zpsz(i) ~= 0
          H = H + log(zpsz(i)) * zpsz(i);
      end
    end
    H=-H/log(factorial(m));


end

