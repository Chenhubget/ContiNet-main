function regcont 
%Comenius University Bratislava, Slovakia, December 2011 
%Auhtors: R.Pasteka, R. Karcol, D.Kusnirak, A. Mojzes
%MAIN FUNCTION
clear all;close all;clc

%main window inicialization   
mainwindow=figure('Name','REGCONT','NumberTitle','off');
bg=get(mainwindow,'Color');
[pic,map]=imread('main_logo.rlg','BackgroundColor',bg);
image(pic);colormap(map);axis image;axis off

set((gca(mainwindow)),'position',[0.001 0.001 1 1])

%main window parameters
set(mainwindow,'Menubar','none','Resize','off','Position',[410,450,560,200])

%navigation bar set up
menu1 = uimenu('Label','Profile data');
        uimenu(menu1,'Label','Open profile data file','Callback',@Open_profile_data_Callback);
        uimenu(menu1,'Label','Display profile data','Callback',@Display_profile_data_Callback);
        uimenu(menu1,'Label','Downward Continuation','Callback',@downward_2D_Callback,'Separator','on');
        uimenu(menu1,'Label','Upward Continuation','Callback',@upward_2D_Callback);
                 
menu2 = uimenu('Label','Grid data');
        uimenu(menu2,'Label','Open grid data file','Callback',@Open_grid_data_Callback);
        uimenu(menu2,'Label','Display grid data','Callback',@Display_grid_data_Callback);
        uimenu(menu2,'Label','Downward Continuation','Callback',@downward_3D_Callback,'Separator','on');
        uimenu(menu2,'Label','Upward Continuation','Callback',@upward_3D_Callback);
                 
menu3 = uimenu('Label','End');
        uimenu(menu3,'Label','About','Callback',@about_Callback);   
        uimenu(menu3,'Label','Quit inside MATLAB','Callback',@matlab_exit_Callback);
        uimenu(menu3,'Label','Quit','Callback','exit','Accelerator','Q');
        
        
%===========================PROFILE DATA===================================
function Open_profile_data_Callback(src,eventdata)
global P

% INPUT OF PROFILE DATA - from 2 columns file (into matrix inpXdT)
P.cdir=pwd;
[P.name,P.path]=uigetfile('*.dat','input of profile data (*.DAT file with 2 columns)');
%change the directory if needed and open the file
cd(P.path);  inpXF = dlmread(P.name);
cd(P.cdir);
% columns of the readed matrix inpXF are separated into vectors X and dT
P.X=inpXF(:,1);  P.F=inpXF(:,2);
% number of readed data-rows is determined - variable m
P.numP = numel(P.X); s = ['  number of readed profile data: ' int2str(P.numP) '                                     (select next items from the "Profile data menu)'];
%profile extrapolation - 15% each edge (to exclude the edge effects during FFT)
[P.eF,P.numeP] = cosxp2prof(P.F,0.15);
P.pointsnextr = numel(P.eF);
%Display data and show info
Display_profile_data_Callback
msgbox(s,'message');


% 1D direct FFT calculation
P.y = fft(P.eF);

% wave-numbers evaluation (saved in the field fk() )
% df is the step of the wave-number
P.dx = P.X(2)-P.X(1);
P.df = 1/(P.dx*(P.pointsnextr-1));
R=rem(P.pointsnextr,2);
if R==0
       P.fk(1:P.pointsnextr/2+1) = 0:P.df:(P.pointsnextr/2)*P.df;
       fk_pom = -P.fk(2:P.pointsnextr/2+1);
       P.fk(P.pointsnextr/2+1:P.pointsnextr) = fk_pom(:,[length(fk_pom):-1:1]);
else
       P.fk(1:round(P.pointsnextr/2))=0:P.df:floor(P.pointsnextr/2)*P.df;
       fk_pom = -P.fk(2:round(P.pointsnextr/2));
       P.fk(round(P.pointsnextr/2)+1:P.pointsnextr) = fk_pom(:,[length(fk_pom):-1:1]);
end;


function Display_profile_data_Callback(src,eventdata)
global orig_fig
global P

    %initial check
    if isempty(P)
        helpdlg('Load profile data first','Wrong selection');
        return
    else
    end

    % plotting the graph of the readed input data
    orig_fig=figure('Name','original data (2D)','NumberTitle','off');
    plot(P.X,P.F);
    % axes properties
    xlabel('x [dist. units]','FontSize',10); ylabel('input function [nT (or) mGal]','FontSize',10);
    title('input - original imported profile data   (select next items from "profile" menu)');
    set(orig_fig,'WindowButtonMotionFcn',@cursorcoordinate)
    

function upward_2D_Callback(src,eventdata)
global fig_ucont_2D ucont_2D prof_UC
global P

%initial check
if isempty(P)
    helpdlg('Load profile data first','Wrong selection');
    return
else
end

% entering of the depth of upward continuation
prompt = {'depth of upward cont. (plus/minus sign not important)'};
answer = inputdlg(prompt,'input parameter');
hinputstr = answer; h = str2double(hinputstr); upwdepth = -abs(h);

 
% calculation of the final upward continued field (for the optimum or selected value of alpha)
   for ji=1:P.pointsnextr
   yt(ji) = P.y(ji)*exp(upwdepth*2*pi*abs(P.fk(ji)));
   end;
   % inverse DFT
   prof_UCe = real(ifft(yt));

%extracting the original length of the profile
   prof_UC = prof_UCe(P.numeP+1:P.pointsnextr-P.numeP);

% plotting the graph of the final upward continued data 

if ishandle(findobj('Name','Upward Continuation(2D)'))
    title_solid=('output - Upward Continued Field, Continuation Height:');
    title_format=strcat(title_solid,'%.1f');title_str=sprintf(title_format, abs(upwdepth));
    set(fig_ucont_2D,'Xdata',P.X,'Ydata',prof_UC)
    title(get(fig_ucont_2D,'Parent'),title_str)
else
    title_solid=('output - Upward Continued Field, Continuation Height:');
    title_format=strcat(title_solid,'%.1f');title_str=sprintf(title_format, abs(upwdepth));
    ucont_2D=figure('Name','Upward Continuation(2D)','NumberTitle','off');
    fig_ucont_2D=plot(P.X,prof_UC); xlabel('x [dist.units]','FontSize',10); ylabel('Upwrard Continuation function [nT/m (or) mGal/m]','FontSize',10);
    title(title_str);
end   
       
out_pos=get(ucont_2D,'OuterPosition');scr=get(0,'ScreenSize');pos=get(ucont_2D,'Position');
set(ucont_2D,'Position',[scr(3)/2 scr(4)/4 pos(3) pos(4)])

% saving the final Upwrard Continued profile data (non-regularized for alpha=0, regularized for alpha<>0)
   for ji=1:P.numP
       savearr(ji,1) = P.X(ji);
       savearr(ji,2) = prof_UC(ji);
   end;
   
     [part1, ext] = strtok(P.name,'.');
     part2='_UC_';
     part3=num2str(abs(upwdepth));
     nametogether=strcat(part1,part2,part3,ext);
     cd(P.path);  
 savefile = fopen(nametogether, 'w');
 fprintf(savefile,'%s %s','X','UC');
 fprintf(savefile, '\n%e %e', savearr');
 fclose(savefile);
     cd(P.cdir);
     
function downward_2D_Callback(src,eventdata)
global optalpha downdepth
global P

%initial check
if isempty(P)
    helpdlg('Load profile data first','Wrong selection');
    return
else
end

% L-NORMS EVALUATION FOR THE REGULARIZED DOWNWARD CONTINUATION (2D)
% initialisation of a help array - prev() -it will contain previous reconstructed field (for previous alpha)
for ji=1:P.pointsnextr
   prev(ji) = 0;
end;

% entering of the depth of downward continuation
prompt = {'depth of downward cont. (plus/minus sign not important)'};
answer = inputdlg(prompt,'input parameter');
hinputstr = answer; h = str2double(hinputstr); downdepth = abs(h);

% input of regular. parameter limits in a window
 prompti = {'start value of regular. parameter (e.g. 1E-5)','end value of regular. parameter (e.g. 1E+20)'};
 titlei = 'norm (downward cont.))';
 linesi = 1; defi = {'1E-5','1E+20'};
 answer = inputdlg(prompti,titlei,linesi,defi);
 opergstr1 = answer(1); alphabegin = str2double(opergstr1);
 opergstr2 = answer(2); alphaend = str2double(opergstr2);
 
% begin of the main cycle - for changing of alpha in a geometrical sequence
% setting the starting value of alpha,  kj is the order number for the increasing alpha
alpha = alphabegin; kj = 1; alpha
% the final value of alpha is set as the limit of the main cycle
while alpha<alphaend
   % evaluation of the regularized downward continuation in spectral domain by means of the Tikhonov filter
   for ji=1:P.pointsnextr
       yt(ji) = P.y(ji)*exp(downdepth*2*pi*abs(P.fk(ji)))/(1+alpha*P.fk(ji)*P.fk(ji)*exp(downdepth*2*pi*abs(P.fk(ji))));   
   end;
   % inverse DFT
   B = real(ifft(yt));
   %L(p)-norms evaluation
   arr_alpha(kj) = alpha;
   norm_function0_5(kj) = norm(B-prev,0.5);
   norm_function0_7(kj) = norm(B-prev,0.7);
   norm_function1(kj) = norm(B-prev,1); 
   norm_function2(kj) = norm(B-prev,2); 
   norm_functionC(kj) = norm(B-prev,inf);
   prev = B;
alpha = 1.1*alpha; kj = kj + 1;
end; 
kj = kj - 2; alpha
% end of the main cycle - for changing of alpha in a geometrical sequence

% finding of local minimum of the C norm function for downward continuation
% the first value of the norm function can not be evaluated correctly, it is taken equal to the second value
norm_functionC(1) = norm_functionC(2);
ndata = kj - 1; position = 0; optalpha = 0; optnorm = 0;
for ji=(ndata-1): -1:3
     if (norm_functionC(ji)<norm_functionC(ji-1)) & (norm_functionC(ji+1)>norm_functionC(ji))
     position = ji; optalpha = arr_alpha(ji); optnorm = norm_functionC(ji);
     end;
end;


% plotting the L norms 
if ishandle(findobj('Name','L norms for the Downward Continuation(2D)'))
   norm_2D=findobj('Name','L norms for the Downward Continuation(2D)');
   cla(gca(norm_2D))
   if optalpha ~= 0
       loglog(gca(norm_2D),arr_alpha,norm_function0_5,arr_alpha,norm_function0_7,arr_alpha,norm_function1,arr_alpha,norm_function2,arr_alpha,norm_functionC,optalpha,optnorm,...
               '--rs','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10)
       xlabel(gca(norm_2D),'log10(alfa) []','FontSize',10); ylabel(gca(norm_2D),'log10{Lnorm[reconstr(j)-reconst(j-1)]} [nT (or) mGal]','FontSize',10);
       legend(gca(norm_2D),'L_0_5 norm','L_0_7 norm','L_1 norm','L_2 norm','C = L_i_n_f norm','Location','South');
       title(gca(norm_2D),{'L norms for the Downward Continuation';'Press ''Z''key to enable picking mode or any key to disable picking mode'});
    else
       loglog(gca(norm_2D),arr_alpha,norm_function0_5,arr_alpha,norm_function0_7,arr_alpha,norm_function1,arr_alpha,norm_function2,arr_alpha,norm_functionC,optalpha,optnorm,...
               '--rs','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10)
       xlabel(gca(norm_2D),'log10(alfa) []','FontSize',10); ylabel(gca(norm_2D),'log10{Lnorm[reconstr(j)-reconst(j-1)]} [nT (or) mGal]','FontSize',10);
       legend(gca(norm_2D),'L_0_5 norm','L_0_7 norm','L_1 norm','L_2 norm','C = L_i_n_f norm','Location','South');
       title(gca(norm_2D),{'L norms for the Downward Continuation, NO LOCAL MINIMUM FOUND';'Press ''Z''key to enable picking mode or any key to disable picking mode'});
    end;
else
    if optalpha ~= 0
       norm_2D=figure('Name','L norms for the Downward Continuation(2D)','NumberTitle','off','Tag','norm_2D');
       loglog(arr_alpha,norm_function0_5,arr_alpha,norm_function0_7,arr_alpha,norm_function1,arr_alpha,norm_function2,arr_alpha,norm_functionC,optalpha,optnorm,...
               '--rs','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10)
       xlabel('log10(alfa) []','FontSize',10); ylabel('log10{Lnorm[reconstr(j)-reconst(j-1)]} [nT (or) mGal]','FontSize',10);
       legend('L_0_5 norm','L_0_7 norm','L_1 norm','L_2 norm','C = L_i_n_f norm','Location','South');
       title({'L norms for the Downward Continuation';'Press ''Z''key to enable picking mode or any key to disable picking mode'});
    else
       norm_2D=figure('Name','L norms for the Downward Continuation(2D)','NumberTitle','off','Tag','norm_2D');
       loglog(arr_alpha,norm_function0_5,arr_alpha,norm_function0_7,arr_alpha,norm_function1,arr_alpha,norm_function2,arr_alpha,norm_functionC,optalpha,optnorm,...
               '--rs','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10,'Tag','norm_2D')
       xlabel('log10(alfa) []','FontSize',10); ylabel('log10{Lnorm[reconstr(j)-reconst(j-1)]} [nT (or) mGal]','FontSize',10);
       legend('L_0_5 norm','L_0_7 norm','L_1 norm','L_2 norm','C = L_i_n_f norm','Location','South');
       title({'L norms for the Downward Continuation, NO LOCAL MINIMUM FOUND';'Press ''Z''key to enable picking mode or any key to disable picking mode'});
    end;
end


out_pos=get(norm_2D,'OuterPosition');scr=get(0,'ScreenSize');pos=get(norm_2D,'Position');
set(norm_2D,'Position',[scr(3)/2-out_pos(3) (scr(4)/4)-20 pos(3) pos(4)])
set(norm_2D,'keypressfcn',@normpick)
downward_calc_2D(optalpha,optalpha);


% saving the norm-functions
  for ji=1:ndata
       savearr(ji,1) = log10(arr_alpha(ji));
       savearr(ji,2) = log10(norm_functionC(ji)); savearr(ji,3) = log10(norm_function2(ji)); savearr(ji,4) = log10(norm_function1(ji)); 
       savearr(ji,5) = log10(norm_function0_7(ji)); savearr(ji,6) = log10(norm_function0_5(ji)); 
  end;
    
 [part1, ext] = strtok(P.name,'.');
 part2='_norms_DC_';
 part3=num2str(downdepth,'%.0f');
 nametogether=strcat(part1,part2,part3,ext);
 cd(P.path);  
 savefile = fopen(nametogether, 'w');
 fprintf(savefile,'%s %s %s %s %s %s','log10(alpha)','log10(C_norm)','log10(L2_norm)','log10(L1_norm)','log10(L0_7_norm)','log10(L0_5_norm)');
 fprintf(savefile, '\n%e %e %e %e %e %e', savearr');
 fclose(savefile);
 cd(P.cdir);
  
if optalpha == arr_alpha(ndata)
   optalpha = 0;
end;

function downward_calc_2D(varargin)
global fig_downcont_2D downcont_2D optalpha prof_DC
global P downdepth

inp=length(varargin);

if length(varargin)==2
    optalpha=varargin{1};pick_alpha=varargin{2};optalpha=pick_alpha;
elseif length(varargin)==1
    optalpha=varargin{1};
end    

% calculation of the final downward continuation field (for the optimum or selected value of alpha)
   for ji=1:P.pointsnextr
       yt(ji) = P.y(ji)*exp(downdepth*2*pi*abs(P.fk(ji)))/(1+optalpha*P.fk(ji)*P.fk(ji)*exp(downdepth*2*pi*abs(P.fk(ji))));

   end;
   % inverse DFT
   prof_DCe = real(ifft(yt));

%extracting the original length of the profile
   prof_DC = prof_DCe(P.numeP+1:P.pointsnextr-P.numeP);

% plotting the graph of the final downward continued data (when the optimum alpha is not equal to the last used alpha)

title_var = '%.3e';
   
if ishandle(findobj('Name','Downward Continuation(2D)'))
    title_solid=('output - regularized Downward Continuation, alpha = ');
    title_format=strcat(title_solid,title_var);title_str=sprintf(title_format, pick_alpha);
    title_format2l=strcat('Continuation Depth:','%.1f');title_str2l=sprintf(title_format2l, abs(downdepth));
    set(fig_downcont_2D,'Xdata',P.X,'Ydata',prof_DC)
    if pick_alpha==0
       title_str=('output - NON-regular. Downward Continuation (alpha = 0)');
    end
    title(get(fig_downcont_2D,'Parent'),{title_str,title_str2l})
else
    title_solid=('output - regularized Downward Continuation, alpha =');
    title_format=strcat(title_solid,title_var);title_str=sprintf(title_format, pick_alpha);
    title_format2l=strcat('Continuation Depth:','%.1f');title_str2l=sprintf(title_format2l, abs(downdepth));
    downcont_2D=figure('Name','Downward Continuation(2D)','NumberTitle','off');
    fig_downcont_2D=plot(P.X,prof_DC); xlabel('x [dist.units]','FontSize',10); ylabel('Downward Continuation function [nT/m (or) mGal/m]','FontSize',10);
    if pick_alpha==0
       title_str=('output - NON-regular. Downward Continuation (alpha = 0)');
    end
    title({title_str,title_str2l});
end   
       

out_pos=get(downcont_2D,'OuterPosition');scr=get(0,'ScreenSize');pos=get(downcont_2D,'Position');
set(downcont_2D,'Position',[scr(3)/2 scr(4)/4-20 pos(3) pos(4)])

% saving the final downward continued data (non-regularized for alpha=0, regularized for alpha<>0)
   for ji=1:P.numP
       savearr(ji,1) = P.X(ji);
       savearr(ji,2) = prof_DC(ji);
   end;
if optalpha ~=0
     [part1, ext] = strtok(P.name,'.');
     part2='_DCreg_';
     part3=num2str(downdepth,'%.0f');
     nametogether=strcat(part1,part2,part3,ext);
     cd(P.path);  
 savefile = fopen(nametogether, 'w');
 fprintf(savefile,'%s %s','X','regDC');
 fprintf(savefile, '\n%e %e', savearr');
 fclose(savefile);
     cd(P.cdir);
else
     [part1, ext] = strtok(P.name,'.');
     part2='_DCnonreg_';
     part3=num2str(downdepth,'%.0f');
     nametogether=strcat(part1,part2,part3,ext);
     cd(P.path);  
 savefile = fopen(nametogether, 'w');
 fprintf(savefile,'%s %s','X','nonregDC');
 fprintf(savefile, '\n%e %e', savearr');
 fclose(savefile);     cd(P.cdir);
end;

        
%=============================GRID DATA====================================
function Open_grid_data_Callback(src,eventdata)
global G

%%%READ ASCII GRID
G.rowsm = 0; G.columnsn = 0; G.minx = 0; G.maxx= 0; G.miny = 0; G.maxy= 0; G.minf = 0; maxf= 0;
[G.name,G.path] = uigetfile('*.grd','input of data (Surfer ASCII *.grid file)');
G.cdir=pwd;cd (G.path)
fid = fopen(G.name,'r');
cd (G.cdir)
% reading of the GS ASCII grid header (rows, columns, G.minx, G.maxx, G.miny, G.maxy, G.minf, maxf)
tline = fgetl(fid); row1 = tline;
tline = fgetl(fid); row2 = tline; [token,rest] = strtok(row2);
G.rowsm = str2double(token); G.columnsn = str2double(rest);
tline = fgetl(fid); row3 = tline; [token,rest] = strtok(row3);
G.minx = str2double(token); G.maxx = str2double(rest);
tline = fgetl(fid); row4 = tline; [token,rest] = strtok(row4);
G.miny = str2double(token); G.maxy = str2double(rest);
tline = fgetl(fid); row5 = tline; [token,rest] = strtok(row5);
G.minf = str2double(token); maxf = str2double(rest);
% reading of the main field of the grid - into the matrix A
[A,count] = fscanf(fid,'%f',[G.rowsm,G.columnsn]);
status = fclose(fid);
G.field = A';
% size of the readed matrix: columnsn x rowsm
[G.columnsn,G.rowsm] = size(A);
s = [' size of readed grid: ' int2str(G.rowsm) ' rows x ' int2str(G.columnsn) ' columns                                    (select next items from the "Grid data" menu)'];

%field axes set up
[mg,ng]=size(G.field);
G.xax=(G.minx:(G.maxx-G.minx)/(ng-1):G.maxx);
G.yax=(G.miny:(G.maxy-G.miny)/(mg-1):G.maxy);


%%% FIELD EXTRAPOLATION - 15% each edge (to exclude the edge effects during FFT)
[G.efield,G.numextrRows,G.numextrCols] = cosxp2(G.field,0.15);% 0.15=15% of extrapolation
%new dimensions of the extrapolated field
[G.erowsm, G.ecolumnsn] = size(G.efield);
%Display data and show info
 Display_grid_data_Callback
 msgbox(s,'message');
 
%%% WAVE-NUMBERS EVALUATION (saved in the fields fkx() and fky()  )
% 2D direct FFT calculation
G.y = fft2(G.efield);

% dfx and dfy are the steps of the wave-numbers
G.dx = abs(G.maxx-G.minx)/(G.columnsn-1); G.dy = abs(G.maxy-G.miny)/(G.rowsm-1);
G.dfx = 1/(G.dx*(G.ecolumnsn-1)); G.dfy = 1/(G.dy*(G.erowsm-1));
R=rem(G.ecolumnsn,2);
if R==0
       G.fkx(1:G.ecolumnsn/2+1) = 0:G.dfx:(G.ecolumnsn/2)*G.dfx;
       fkx_pom = -G.fkx(2:G.ecolumnsn/2+1);
       G.fkx(G.ecolumnsn/2+1:G.ecolumnsn) = fkx_pom(:,[length(fkx_pom):-1:1]);
else
       G.fkx(1:round(G.ecolumnsn/2))=0:G.dfx:floor(G.ecolumnsn/2)*G.dfx;
       fkx_pom = -G.fkx(2:round(G.ecolumnsn/2));
       G.fkx(round(G.ecolumnsn/2)+1:G.ecolumnsn) = fkx_pom(:,[length(fkx_pom):-1:1]);
end;
R=rem(G.erowsm,2);
if R==0
       G.fky(1:G.erowsm/2+1) = 0:G.dfy:(G.erowsm/2)*G.dfy;
       fky_pom = -G.fky(2:G.erowsm/2+1);
       G.fky(G.erowsm/2+1:G.erowsm) = fky_pom(:,[length(fky_pom):-1:1]);
else
       G.fky(1:round(G.erowsm/2))=0:G.dfy:floor(G.erowsm/2)*G.dfy;
       fky_pom = -G.fky(2:round(G.erowsm/2));
       G.fky(round(G.erowsm/2)+1:G.erowsm) = fky_pom(:,[length(fky_pom):-1:1]);
end;  

function Display_grid_data_Callback(src,eventdata)
global G

%initial check
if isempty(G)
    helpdlg('Load grid data first','Wrong selection');
    return
else
end

% plotting the contour map of the readed input grid data
orig_fig_g=figure('Name','original data (3D)','NumberTitle','off');
pcolor(G.xax,G.yax,G.field); cb=colorbar; shading interp;
xlabel ('x[dist.units]','FontSize',10); ylabel('y[dist.units]','FontSize',10);
title('input - original imported grid data')
hold on;contour(G.xax,G.yax,G.field,'k');hold off;
colormap(gca(orig_fig_g),colscale)
set(orig_fig_g,'WindowButtonMotionFcn',@cursorcoordinate)
set(get(cb,'ylabel'),'string','[nT (or) mGal]')
    
function upward_3D_Callback(src,eventdata)
global up3DH G

%initial check
if isempty(G)
    helpdlg('Load grid data first','Wrong selection');
    return
else
end

% entering of the depth of upward continuation
 prompt = {'depth of upward cont. (plus/minus sign not important)'};   
 answer = inputdlg(prompt,'input parameter');
 hinputstr = answer; h = str2double(hinputstr); upwdepth = -abs(h);

% calculation of the upward continued field (no regularization)
   for j=1:G.erowsm
       for k=1:G.ecolumnsn
           yt(j,k) = G.y(j,k)*exp(upwdepth*2*pi*sqrt(G.fkx(k)*G.fkx(k)+G.fky(j)*G.fky(j)));
       end
   end;
   % inverse DFT
   finalUC = real(ifft2(yt));

%exctracting back to the original size of the input grid
finalUC2 = finalUC(G.numextrRows+1:G.erowsm-G.numextrRows,G.numextrCols+1:G.ecolumnsn-G.numextrCols);

% saving the final upward continued data (when the optimum alpha is not equal to the last used alpha)
% EXPORT OF THE GRID FILE IN THE SURFER ASCII *.GRD FORMAT

minv = min(min(finalUC2)); maxv=max(max(finalUC2));

cd(G.path)   

%creating of the output file name

[part1, ext] = strtok(G.name,'.');
part2='_UC_';
part3=num2str(abs(upwdepth),'%.0f');
nametogether=strcat(part1,part2,part3,ext);
fid = fopen(nametogether,'w');

fprintf(fid,'%s\n','DSAA');
fprintf(fid,'%3.0f %3.0f\n',G.columnsn,G.rowsm);
fprintf(fid,'%f %f\n',G.minx,G.maxx);fprintf(fid,'%f %f\n',G.miny,G.maxy);fprintf(fid,'%e %e\n',minv,maxv);
fprintf(fid,'%e ',finalUC2');
status = fclose(fid);
cd(G.cdir)

%Plot the result

if ishandle(findobj('Name','Upward Continuation(3D)'))
    title_solid=('output - Upward Continued Field, Continuation Height:');
    title_format=strcat(title_solid,'%.1f');title_str=sprintf(title_format, abs(upwdepth));
    set(up3DH.pc,'Cdata',finalUC2);set(up3DH.cont,'Zdata',finalUC2)
    title(get(up3DH.pc,'Parent'),title_str)
    colormap(gca(up3DH.fig),colscale)
else
    title_solid=('output - Upward Continued Field, Continuation Height:');
    title_format=strcat(title_solid,'%.1f');title_str=sprintf(title_format, abs(upwdepth));
    up3D=figure('Name','Upward Continuation(3D)','NumberTitle','off');
    up3DH = guihandles(up3D); up3DH.fig=up3D;
    up3DH.pc=pcolor(G.xax,G.yax,finalUC2);up3DH.cb=colorbar;shading interp 
    xlabel('x [dist. units]','FontSize',10); ylabel('y [dist. units]','FontSize',10);
    title(title_str);hold on;[C,up3DH.cont]=contour(G.xax,G.yax,finalUC2,'k');hold off;
    colormap(gca(up3DH.fig),colscale)
    set(up3DH.fig,'WindowButtonMotionFcn',@cursorcoordinate)
end   

%store handle structure
guidata(gcbo,up3DH)

out_pos=get(up3DH.fig,'OuterPosition');scr=get(0,'ScreenSize');pos=get(up3DH.fig,'Position');
set(up3DH.fig,'Position',[scr(3)/2 (scr(4)/4)-20 pos(3) pos(4)])
set(get(up3DH.cb,'ylabel'),'string','[nT (or) mGal]')

function downward_3D_Callback(src,eventdata)
global optalphag G downdepth

%initial check
if isempty(G)
    helpdlg('Load grid data first','Wrong selection');
    return
end

% entering of the depth of downward continuation
prompt = {'depth of downward cont. (plus/minus sign not important)'};   
answer = inputdlg(prompt,'input parameter');
hinputstr = answer; h = str2double(hinputstr); downdepth = abs(h);

% input of regular. parameter limits in a window
inform = 'evaluation of the norm (downward cont.)'; inform
prompti = {'start value of regular. parameter (e.g. 1E0)','end value of regular. parameter (e.g. 1E+20)'};
titlei = 'norm (Downward Continuation)';
linesi = 1; defi = {'1E0','1E+20'};
answer = inputdlg(prompti,titlei,linesi,defi);
opergstr1 = answer(1); alphabeging3 = str2double(opergstr1);
opergstr2 = answer(2); alphaendg3 = str2double(opergstr2);
  
%MAIN CYCLE OF EVALUATION OF THE NORM FOR THE REGULARIZED DOWNWARD CONTINUATION (GRID)
% initialisation of a help matrix - prevji() -it will contain previous reconstructed field (for previous alpha)
prevji = zeros(G.erowsm,G.ecolumnsn);
% begin of the main cycle - for changing of alpha in a geometrical sequence
% setting the starting value of alpha,  kj is the order number for the increasing alpha
alpha = alphabeging3; kj = 1; alpha
% the final value of alpha is set as the limit of the main cycle
while alpha<alphaendg3
%alpha
   % evaluation of the downward continuation in spectral domain by means of the Tikhonov filter application
   for j=1:G.erowsm
       for k=1:G.ecolumnsn  
       yt(j,k) = G.y(j,k)*exp(downdepth*2*pi*sqrt(G.fkx(k)*G.fkx(k)+G.fky(j)*G.fky(j)))/...
                 (1+alpha*(G.fkx(k)*G.fkx(k)+G.fky(j)*G.fky(j))*exp(downdepth*2*pi*sqrt(G.fkx(k)*G.fkx(k)+G.fky(j)*G.fky(j))));	   
       end;
   end;
   % inverse DFT
   B = real(ifft2(yt));
   
% Cnorm-function calculation
   maxdiffer = 0;
   for j=2:G.erowsm-1
     for k=2:G.ecolumnsn-1
       if abs(B(j,k)-prevji(j,k))>maxdiffer
          maxdiffer = abs(B(j,k)-prevji(j,k)); 
       end;
     end;
   end;
   arr_alpha(kj) = alpha;
   norm_function(kj) = maxdiffer;
   %L(p)-norms evaluation
   norm_function1(kj) = norm(B-prevji,1); norm_function2(kj) = norm(B-prevji,2);
   norm_functionC(kj) = norm(B-prevji,inf); norm_function_Fro(kj) = norm(B-prevji,'fro');
   % filling of the help matrix prevji() with the last actual solution (for last actual alpha)
   prevji = B;
% increasing of alpha (in a geometrical sequence, with a quocient of 1.1)
alpha = 1.1*alpha; kj = kj + 1;
end; 
kj = kj - 2; alpha
% end of the main cycle - for changing of alpha in a geometrical sequence

% finding of local minimum of the C norm function
% the first value of the norm function can not be evaluated correctly, it is taken equal to the second value
norm_function(1) = norm_function(2);
ndata = kj - 1; position = 0; optalphag = 0; optnorm = 0;
for ji=(ndata-1): -1:3
     if (norm_function(ji)<norm_function(ji-1)) & (norm_function(ji+1)>norm_function(ji))
     position = ji;
     optalphag = arr_alpha(ji);
     optnorm = norm_function(ji);
     end;
end;

% plotting the L-norms for downward continuation
if ishandle(findobj('Name','L norms for the Downward Continuation(3D)'))
   norm_3D=findobj('Name','L norms for the Downward Continuation(3D)');
   cla(gca(norm_3D))
   if  optalphag ~= 0
       loglog(gca(norm_3D),arr_alpha,norm_function1,arr_alpha,norm_function2,arr_alpha,norm_function,optalphag,optnorm,...
               '--rs','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10)
       xlabel(gca(norm_3D),'log10(alfa) []','FontSize',10); ylabel(gca(norm_3D),'log10Cnorm{reconstr(j)]-reconst(j-1)} [nT (or) mGal]','FontSize',10);
       legend(gca(norm_3D),'L_1 norm','L_2 norm','C = L_i_n_f norm','Location','South');
       title(gca(norm_3D),{'L norms for the Downward Continuation';'Press ''Z''key to enable picking mode or any key to disable picking mode'});
    else
       loglog(gca(norm_3D),arr_alpha,norm_function1,arr_alpha,norm_function2,arr_alpha,norm_function,optalphag,optnorm,...
               '--rs','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10)
       xlabel(gca(norm_3D),'log10(alfa) []','FontSize',10); ylabel(gca(norm_3D),'log10Cnorm{reconstr(j)]-reconst(j-1)} [nT (or) mGal]','FontSize',10);
       legend(gca(norm_3D),'L_1 norm','L_2 norm','C = L_i_n_f norm','Location','South');
       title(gca(norm_3D),{'L norms for the Downward Continuation, NO LOCAL MINIMUM FOUND';'Press ''Z''key to enable picking mode or any key to disable picking mode'});
    end;
else
    if optalphag ~= 0
       norm_3D=figure('Name','L norms for the Downward Continuation(3D)','NumberTitle','off');
       loglog(arr_alpha,norm_function1,arr_alpha,norm_function2,arr_alpha,norm_function,optalphag,optnorm,...
               '--rs','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10)
       xlabel('log10(alfa) []','FontSize',10); ylabel('log10Cnorm{reconstr(j)]-reconst(j-1)} [nT (or) mGal]','FontSize',10);
       legend('L_1 norm','L_2 norm','C = L_i_n_f norm','Location','South');
       title({'L norms for the Downward Continuation';'Press ''Z''key to enable picking mode or any key to disable picking mode'});
    else
       norm_3D=figure('Name','L norms for the Downward Continuation(3D)','NumberTitle','off');
       loglog(arr_alpha,norm_function1,arr_alpha,norm_function2,arr_alpha,norm_function,optalphag,optnorm,...
               '--rs','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',10)
       xlabel('log10(alfa) []','FontSize',10); ylabel('log10Cnorm{reconstr(j)]-reconst(j-1)} [nT (or) mGal]','FontSize',10);
       legend('L_1 norm','L_2 norm','C = L_i_n_f norm','Location','South');
       title({'L norms for the Downward Continuation, NO LOCAL MINIMUM FOUND';'Press ''Z''key to enable picking mode or any key to disable picking mode'});
    end;
end

% set up the window position and calladditional functions
out_pos=get(norm_3D,'OuterPosition');scr=get(0,'ScreenSize');pos=get(norm_3D,'Position');
set(norm_3D,'Position',[scr(3)/2-out_pos(3) (scr(4)/4)-20 pos(3) pos(4)])

set(norm_3D,'keypressfcn',@normpick)

downward_calc_3D(optalphag,optalphag);

% saving the norm-functions
  for ji=1:ndata
       savearr(ji,1) = log10(arr_alpha(ji));
       savearr(ji,2) = log10(norm_function(ji)); savearr(ji,3) = log10(norm_function2(ji)); savearr(ji,4) = log10(norm_function1(ji)); 
  end;
  
[part1, ext] = strtok(G.name,'.');
part2='_norms_DC_';
part3=num2str(downdepth,'%.0f');
nametogether=strcat(part1,part2,part3,'.dat');

cd (G.path)
savefile = fopen(nametogether, 'w');
fprintf(savefile,'%s %s %s %s %s %s','log10(alpha)','log10(C_norm)','log10(L2_norm)','log10(L1_norm)');
 fprintf(savefile, '\n%e %e %e %e', savearr');
fclose(savefile);
cd (G.cdir)
     
function downward_calc_3D(varargin)
global fig_down_3D down_3D fig_down_3D_cont optalphag G
global downdepth
 
if length(varargin)==2
    optalphag=varargin{1};pick_alpha=varargin{2};optalphag=pick_alpha;
elseif length(varargin)==1
    optalphag=varargin{1};
end    


% calculation of the downward continuation field (for the optimum or selected value of alpha)
   for j=1:G.erowsm
       for k=1:G.ecolumnsn
       yt(j,k) = G.y(j,k)*exp(downdepth*2*pi*sqrt(G.fkx(k)*G.fkx(k)+G.fky(j)*G.fky(j)))/...
                 (1+optalphag*(G.fkx(k)*G.fkx(k)+G.fky(j)*G.fky(j))*exp(downdepth*2*pi*sqrt(G.fkx(k)*G.fkx(k)+G.fky(j)*G.fky(j))));
	   end;
   end;
   % inverse DFT
   finalDC = real(ifft2(yt));

%exctracting back to the original size of the input grid
finalDC2 = finalDC(G.numextrRows+1:G.erowsm-G.numextrRows,G.numextrCols+1:G.ecolumnsn-G.numextrCols);

% saving the final downward continued data (when the optimum alpha is not equal to the last used alpha)
% EXPORT OF THE GRID FILE IN THE SURFER ASCII *.GRD FORMAT

minv = min(min(finalDC2)); maxv=max(max(finalDC2));

cd(G.path)   
if optalphag ~= 0
    %creating of the outputfile name - in the first part using the name of the input grid
    [part1, ext] = strtok(G.name,'.');
    part2='_DCreg_';
    part3=num2str(downdepth,'%.0f');
    nametogether=strcat(part1,part2,part3,ext);
    fid = fopen(nametogether,'w');
 else
    %creating of the outputfile name - in the first part using the name of the input grid
    [part1, ext] = strtok(G.name,'.');
    part2='_DCnonreg_';
    part3=num2str(downdepth,'%.0f');
    nametogether=strcat(part1,part2,part3,ext);
    fid = fopen(nametogether,'w');
end
fprintf(fid,'%s\n','DSAA');
fprintf(fid,'%3.0f %3.0f\n',G.columnsn,G.rowsm);
fprintf(fid,'%f %f\n',G.minx,G.maxx);fprintf(fid,'%f %f\n',G.miny,G.maxy);fprintf(fid,'%e %e\n',minv,maxv);
fprintf(fid,'%e ',finalDC2');
status = fclose(fid);
cd(G.cdir)


% plotting the contour map of the downward continuation
if length(varargin)==2
   
    title_var = '%.3e';
   
   if ishandle(findobj('Name','Downward Continuation(3D)'))
        title_solid=('Regularized downward continuation, alpha = ');
        title_format=strcat(title_solid,title_var);title_str=sprintf(title_format, pick_alpha);
        title_format2l=strcat('Continuation depth:','%.1f');title_str2l=sprintf(title_format2l,abs(downdepth));
        set(fig_down_3D,'Cdata',finalDC2);set(fig_down_3D_cont,'Zdata',finalDC2)
        if pick_alpha==0
           title_str=('NON-regular. downward continuation (alpha = 0)');
        end
        title(get(fig_down_3D,'Parent'),{title_str,title_str2l})
        colormap(gca(down_3D),colscale);cb=colorbar('peer',gca(down_3D));
   else
        title_solid=('Regularized downward continuation, alpha = ');
        title_format=strcat(title_solid,title_var);title_str=sprintf(title_format, pick_alpha);
        title_format2l=strcat('Continuation depth:','%.1f');title_str2l=sprintf(title_format2l,abs(downdepth));
        down_3D=figure('Name','Downward Continuation(3D)','NumberTitle','off');
        fig_down_3D=pcolor(G.xax,G.yax,finalDC2);cb=colorbar; shading interp 
        xlabel('x [dist. units]','FontSize',10); ylabel('y [dist. units]','FontSize',10);
        if pick_alpha==0
           title_str=('NON-regular. downward continuation (alpha = 0)');
        end
        title({title_str,title_str2l});hold on;[C,fig_down_3D_cont]=contour(G.xax,G.yax,finalDC2,'k');hold off;
        colormap(gca(down_3D),colscale)
        set(down_3D,'WindowButtonMotionFcn',@cursorcoordinate)
   end   

end

% figure position settings
out_pos=get(down_3D,'OuterPosition');scr=get(0,'ScreenSize');pos=get(down_3D,'Position');
set(down_3D,'Position',[scr(3)/2 (scr(4)/4)-20 pos(3) pos(4)])
set(get(cb,'ylabel'),'string','[nT (or) mGal]')


%4. ABOUT~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

function about_Callback(src,eventdata)
% ABOUT
msgbox('                           REGCONT, 2011                                      Comenius University Bratislava, Slovakia                                                 Authors: R.Pasteka, R. Karcol, D.Kusnirak, A. Mojzes    pasteka@fns.uniba.sk','About')

function matlab_exit_Callback(src,eventdata)
clear all;close all;clc


 
%=====================ADDITIONAL FUNCTIONS=================================

function [g,PointExt] = cosxp2prof(T,perc)
%Expand profile using cosine taper mirror function, 'perc' (%) expansion. 
%where perc is a decimal that represents the expansion %, e.g. 0.15 --> 15% 
%g - final extrapolated field; PointExt - number of the added points on one edge
%Written by: Daniela Gerovska and Marcos Arauzo-Bravo (27/February/2003) 
if ~exist('perc')
    perc=0.1;   % default exapnsion is 10%
end
perc = 1+perc;

NumPoints = numel(T);
%Deciding NumPointsExt NumColExt
NumPointsExt=Eve10Ext(NumPoints,perc);
g=T;
%Extention of the grid with half a cosine function
% Cosine taper right side
lim=NumPointsExt-NumPoints; rlamb=lim+1; arg=pi/rlamb;
s=0.5*(g(NumPoints)+g(1)); r=0.5*(g(NumPoints)-g(1));
for k=1:lim
    ge=s+r*cos(arg*k);
    g(NumPoints+k)=ge;
end
% Cosine taper left side
g = flipud(g);
lim=NumPointsExt-NumPoints; rlamb=lim+1; arg=pi/rlamb;
s=0.5*(g(NumPointsExt)+g(lim+1)); r=0.5*(g(NumPointsExt)-g(lim+1));
for k=1:lim
    ge=s+r*cos(arg*k);
    g(NumPointsExt+k)=ge;
end
g = flipud(g);
PointExt = NumPointsExt - NumPoints;

function [g,RowExt,ColExt] = cosxp2(T,perc)
%Expand grid using cosine taper mirror function, 'perc' (%) expansion. 
%where perc is a decimal that represents the expansion %, e.g. 0.15 --> 15% 
%g - final extrapolated field; RowExt - number of the added rows on one edge, ColExt - the same for columns
%Written by: Daniela Gerovska and Marcos Arauzo-Bravo (27/February/2003) 

if ~exist('perc')
    perc=0.1;   % default exapnsion is 10%
end
perc = 1+perc;

[NumRow,NumCol]=size(T);
%Deciding NumRowExt NumColExt
NumRowExt=Eve10Ext(NumRow,perc);
NumColExt=Eve10Ext(NumCol,perc);
g=T;
%Extention of the grid with half a cosine function
% Cosine taper right side
lim=NumColExt-NumCol; rlamb=lim+1; arg=pi/rlamb;
s=0.5*(g(:,NumCol)+g(:,1)); r=0.5*(g(:,NumCol)-g(:,1));
for k=1:lim
    ge=s+r*cos(arg*k);
    g(:,NumCol+k)=ge;
end
% Cosine taper left side
g = fliplr(g);
lim=NumColExt-NumCol; rlamb=lim+1; arg=pi/rlamb;
s=0.5*(g(:,NumColExt)+g(:,lim+1)); r=0.5*(g(:,NumColExt)-g(:,lim+1));
for k=1:lim
    ge=s+r*cos(arg*k);
    g(:,NumColExt+k)=ge;
end
g = fliplr(g);
% Cosine taper upper side
g=g';
lim=NumRowExt-NumRow; rlamb=lim+1; arg=pi/rlamb;
s=0.5*(g(:,NumRow)+g(:,1)); r=0.5*(g(:,NumRow)-g(:,1));
for k=1:lim
    ge=s+r*cos(arg*k);
    g(:,NumRow+k)=ge;
end
g=g';
% Cosine taper lower side
g = flipud(g); g=g';
lim=NumRowExt-NumRow; rlamb=lim+1; arg=pi/rlamb;
s=0.5*(g(:,NumRowExt)+g(:,lim+1)); r=0.5*(g(:,NumRowExt)-g(:,lim+1));
for k=1:lim
    ge=s+r*cos(arg*k);
    g(:,NumRowExt+k)=ge;
end
g=g'; g = flipud(g);
RowExt = NumRowExt - NumRow; ColExt = NumColExt - NumCol;

function NumEveExt=Eve10Ext(Num,perc)
%Even 'perc'(%) extension of the number Num
%Written by: Daniela Gerovska and Marcos Arauzo-Bravo (27/February/2003) 
 NumExt=perc*Num;
NumExtRou=round(NumExt);
if rem(NumExtRou,2)==1 
  DifPlu=abs(NumExt-(NumExtRou+1));
  DifMin=abs(NumExt-(NumExtRou-1));
    if DifPlu>DifMin
       NumEveExt=NumExtRou-1;
    else 
       NumEveExt=NumExtRou+1;
    end%if DifPlu>DifMin  
else
   NumEveExt=NumExtRou;   
end %if rem(NumExtRou,2)==1  

function cursorcoordinate(src,eventdata)

% find axes associated with calling window
hax = gca(src);     

% Axes limits
xlims = get(hax, 'XLim');
ylims = get(hax, 'YLim');

% format for x coordinate
if max(abs(xlims))>999999       %exponencial format definition for values over 10^6
   xformat_str          = 'X: %.2e,';
elseif max(abs(xlims))<0.1      %more precise format definition for values under 0.1
   xformat_str          = 'X: %.4f,';
else                            %standard format with 2 decimals
   xformat_str          = 'X: %.2f,';
end

% format for y coordinate
if max(abs(ylims))>999999
   yformat_str          = ' Y: %.2e';
elseif max(abs(ylims))<0.1
   yformat_str          = ' Y: %.4f';
else
   yformat_str          = ' Y: %.2f';
end

format_str=strcat(xformat_str,yformat_str);

% initial text string
curr_position   = get(hax, 'CurrentPoint');
position        = [curr_position(1, 1) curr_position(1, 2)];
position_string = sprintf(format_str, position); %position string as a text


% do not update if current mouse position is out of axes 
in_bounds = (position(1) >= xlims(1)) && ...
            (position(1) <= xlims(2)) && ...
            (position(2) >= ylims(1)) && ...
            (position(2) <= ylims(2));

if (~in_bounds)  
    return
end        

user_data = get(src, 'UserData'); 

%info text inicialization during the first callback call and its save 
%to the axes handle as as a UserData. Units were set to normalized to
%keep the relative position with axes

if ((~isfield(user_data, 'cursorcoordinate')) || ~ishandle(user_data.cursorcoordinate))

    user_data.cursorcoordinate = text(0, 0, 'blabla','Units','normalized','Parent',hax);

    set(src, 'UserData', user_data); % save to axes
end


%'online' text update
set(user_data.cursorcoordinate, 'Position', [0 -0.1],...
                           'String', position_string,...
                           'VerticalAlignment', 'bottom',...
                           'HorizontalAlignment', 'left',...
                           'Parent',hax);
    
function normpick(src,eventdata)
global optalpha optalphag

fig=src; 
cur_char=get(src,'CurrentKey');
set(fig,'WindowButtonMotionFcn',@cursorcoordinate)

if cur_char=='z' || cur_char=='Z'
   
    for i=1:1000
       set(fig,'pointer','Fullcross')
       k=waitforbuttonpress;
       
       if k==0
           button=get(fig,'SelectionType');
           cor=get(gca(fig),'CurrentPoint');
           corx=cor(1,1);cory=cor(1,2);
           cor_i(i,1:3)=cor(1,:);
           if strcmp(button, 'alt')
              corx=0;
              if i==1
                 cor_i(1,1:3)=0;
              end
              cor_i(i,1:3)=cor_i(i-1,1:3);
           end
           set(gca(fig),'NextPlot','add')
           scatter(gca(fig),cor_i(:,1),cor_i(:,2),5,'+k')
            switch get(src,'Name')
                case ('L norms for the Downward Continuation(2D)')
                    downward_calc_2D(optalpha,corx);
                case ('L norms for the Downward Continuation(3D)')
                    downward_calc_3D(optalphag,corx);

            end
            
       elseif k==1
           set(fig,'pointer','arrow')
           refresh(fig)
           break
           return
       end
   end
end

function [scale]=colscale()

% clr file import
fid = fopen('REGCONT1_0.clr');
fscanf(fid,'%s',1);
fscanf(fid,'%f',2);
C = fscanf(fid,'%f');
fclose(fid);

%colormap conversion to the matlab format
D = (reshape(C,4,numel(C)/4))';
scale=((D(:,2:4))/255);
        
        