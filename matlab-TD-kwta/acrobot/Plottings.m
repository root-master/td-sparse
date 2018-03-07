nMeshp = 60;
nMeshv = 60;
grafica = true;
task = 'acrobot';
set(0,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontName','Times New Roman');
set(0,'DefaultAxesFontSize',18);
set(0,'DefaultTextFontSize',18);
set(0, 'defaultTextInterpreter', 'latex');


for i=1:4,
   if i==1, functionApproximator='kwtaNN';
   elseif i==2, functionApproximator='linearNN';
   elseif i==3, functionApproximator='regularBPNN';
   elseif i==4, functionApproximator='Qtable';
   end
   filename = strcat('Results-Jan-1/',functionApproximator,'-nMesh20','.mat');  
   load(filename);
   
                
   h = figure;   
        subplot(2,1,1);
        plot(data.meanDeltaForEpisode)
        
        axx = gca;
        axx.XTickLabel = {0,50000,100000,150000,200000};
        ylabel('mean(\delta) in episode')
        title(['Performance for ',functionApproximator])
        subplot(2,1,2); 
        plot(data.steps)
        xlabel('Episode number')
        ylabel('number of steps')
        axx = gca;
        axx.XTickLabel = {0,50000,100000,150000,200000};
        figname = ['figures/',task,'-',functionApproximator,'-performance'];        
        print(h,figname,'-depsc')

        
    h=figure;
    
        subplot(2,1,1);
        xlabel('Episode number')
        plot(data.meanDeltaForEpisodeTest)
        axx = gca;
        axx.XTickLabel = {0,50000,100000,150000,200000};
        ylabel('mean(\delta) in episode')
        title(['Test with frozen weights for ',functionApproximator])    
        subplot(2,1,2); 
        plot(data.stepsTest)
        xlabel('Episode number')
        ylabel('number of steps')
        axx = gca;
        axx.XTickLabel = {0,50000,100000,150000,200000};
        figname = ['figures/',task,'-',functionApproximator,'-frozen-test'];        
        print(h,figname,'-depsc')

end
