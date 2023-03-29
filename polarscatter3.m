function polarscatter3(theta,phi,rho, varargin )
    maxr=1;
    if nargin>2, maxr = max(ceil(max(rho)),1); end

    %% Plot axis
    clrh = 0;
    if ~ishold, clf, clrh = 1; end 
    hold on;
    for r = maxr/5:maxr/5:maxr
        [tax_x,tks_y] = pol2cart(linspace(0,2*pi,101),r);
        tax_z = zeros(length(tax_x),1); 
        pax_x = tax_z; pax_y = tax_x; pax_z = tks_y;
                
        if r==maxr
            plot3(pax_x,pax_y,pax_z,'-','Color','#606060','LineWidth',1)
            patch(pax_x,pax_y,pax_z,'w','FaceAlpha',0.6)
            plot3(tax_x,tks_y,tax_z,'-','Color','#606060','LineWidth',1)
            patch(tax_x,tks_y,tax_z,'w','FaceAlpha',0.6)
        else
            plot3(pax_x,pax_y,pax_z,'-','Color','#AAAAAA','LineWidth',0.1)
            plot3(tax_x,tks_y,tax_z,'-','Color','#AAAAAA','LineWidth',0.1) 
        end
    end
    plot3(-1:1,[0,0,0],[0,0,0],'-','Color','#000000','LineWidth',1)
    plot3([0,0,0],-1:1,[0,0,0],'-','Color','#000000','LineWidth',1)
    plot3([0,0,0],[0,0,0],-1:1,'-','Color','#000000','LineWidth',1)
    
    lim = maxr+0.1;
    xlim([-lim lim]), ylim([-lim lim]), zlim([-lim lim])
    set(gca,'xtick',[]),set(gca,'ytick',[]),set(gca,'ztick',[]);
    axis off;
    view([135,45])

    %% Plot ticks
        r = 1;
     for d = 0:pi/10:2*pi
        [tks_x,tks_y] = pol2cart([d d],[r r+0.05]); 
        cero = [0 0];
        plot3(tks_x,tks_y,cero,'-','Color','#606060','LineWidth',1)
        plot3(cero,tks_x,tks_y,'-','Color','#606060','LineWidth',1)
     end
     
     %%Plot scater of points
     if nargin <2
         return;
     end
     [x,y,z] = sph2cart(theta,phi,rho);
     if nargin<4
         scatter3(x,y,z)
     else
         scatter3(x,y,z,varargin{:})
     end
     
     if clrh, hold off; end
end

%Juan Esteban Villegas (2023). polarscatter3 (https://www.mathworks.com/matlabcentral/fileexchange/92910-polarscatter3), MATLAB Central File Exchange. Retrieved March 20, 2023.
