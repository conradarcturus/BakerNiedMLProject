function displaytrips()
% Makes some figures about feature data
%
% A. Conrad Nied
%
% 2013-11-25

datadir = '/projects/onebusaway/BakerNiedMLProject/data/routefeatures';
imdir = '/projects/onebusaway/BakerNiedMLProject/figures';
service = 'intercitytransit';
route = 'route13';
filename = sprintf('%s/%s_%s_allfeats.txt', datadir, service, route);
[dist,distold,lat,lon,timeglobal,dayofweek,days,time,tripid,dev] = fetchdata(filename);
% [dist,~,~,~,~,~,days,~,tripid,dev] = fetchdata(filename);

% Get Unique days
N = length(dist);
m = 0;
daynum = [];
tripnum = [];
sel = [];
daysu = unique(days);
tripidu = unique(tripid);

ints = [];
for d = daysu'
    for t = tripidu'
        subset = (days == d) & (tripid == t);
        if(sum(subset))
            m = m + 1;
            daynum(m) = d;
            tripnum(m) = t;
            sel(m, :) = subset;
        end
        ints = [ints; diff(time(find(subset)))];
    end % Trips
end % Days

cmap = jet(m);

figure(100)
clf
histints = hist(ints, 0:10:180)
bar(0:10:180, histints)
title('Interval between bus updates')

figure(1)
clf
for i = 1:m
    line(distold(find(sel(i, :))), dist(find(sel(i, :))), 'Color', cmap(i, :));
end
axis equal
ylabel('Distance computed by latitude, longitude, and route shape (meters)')
xlabel('Distance provided by Puget Sound data feed (meters)')
title({'Comparison of Distance Along Trip Computations', 'Colors represent different trips'})
frame = getframe(gcf);
filename = sprintf('%s/%s_%s_distmetric_comparison.png', imdir, service, route);
imwrite(frame.cdata, filename);

figure(2)
clf
hold on
% for i = 1:m
    %plot(lon(find(sel(i, :))), lat(find(sel(i, :))), 'Color', cmap(i, :));
%     scatter(lon(find(sel(i, :))), lat(find(sel(i, :))), ones(sum(sel(i, :)), 1) * 30, cmap(i, :), 'x');
% end
scatter(lon, lat, ones(N, 1) * 30, dist, 'x');
axis equal
xlabel('Longitude (degrees)')
ylabel('Latitude (degrees)')
title({'Comparison of Distance Along Trip Computations', 'Color shows distance computed by latitude, longitude, and route shape'})
frame = getframe(gcf);
filename = sprintf('%s/%s_%s_distmetric_comparison_distroute.png', imdir, service, route);
imwrite(frame.cdata, filename);

figure(3)
clf
hold on
scatter(lon, lat, ones(N, 1) * 30, distold, 'x');
axis equal
xlabel('Longitude (degrees)')
ylabel('Latitude (degrees)')
title({'Comparison of Distance Along Trip Computations', 'Color shows distance provided by Puget Sound data feed'})
frame = getframe(gcf);
filename = sprintf('%s/%s_%s_distmetric_comparison_distoldroute.png', imdir, service, route);
imwrite(frame.cdata, filename);

figure(4)
clf
hold on
for i = 1:m
    routesel = find(sel(i, :));
    [~, ii] = unique(dist(routesel));
    routesel = routesel(ii);
    plot(dist(routesel), dev(routesel), 'Color', cmap(i, :), 'LineWidth', 2);
end
xlim([min(dist), max(dist)])
ylim([min(dev), max(dev)])
xlabel('Distance (meters)')
ylabel('Schedule Deviation (seconds)')
frame = getframe(gcf);
filename = sprintf('%s/%s_%s_tripInDay_interp_dist_dev.png', imdir, service, route);
imwrite(frame.cdata, filename);


figure(5)
clf
hold on
for i = 1:m
    routesel = find(sel(i, :));
    distpred = 0:100:9000;
    [~, ii] = unique(dist(routesel));
    routesel = routesel(ii);
    try
    devpred = interp1(dist(routesel), dev(routesel), distpred);
    if(~sum(devpred > max(dev) | devpred < min(dev)))
        plot(distpred, devpred, 'Color', cmap(i, :), 'LineWidth', 2);
    end
    end
end
xlim([min(dist), max(dist)])
ylim([min(dev), max(dev)])
xlabel('Distance (meters)')
ylabel('Schedule Deviation (seconds)')
% title({'Comparison of Distance Along Trip Computations', 'Color shows distance provided by Puget Sound data feed'})
frame = getframe(gcf);
filename = sprintf('%s/%s_%s_tripInDay_interp_dist_dev.png', imdir, service, route);
imwrite(frame.cdata, filename);

figure(6)
clf
hold on
% dcmap = jet(length(daysu));
% for i_d = 1:length(daysu)
%     d = daysu(i_d);
%     plot(time(days == d)/3600, dev(days == d), 'Color', dcmap(i_d, :));
% end
% j = 0;
% ccmap = jet(10);
for i = 1:m
%     routetime = time(find(sel(i, :)));
%     routetime = routetime - min(routetime);
    routesel = find(sel(i, :));
%     if(sum(days(routesel) == 1) && j < 1000)
%         j = j + 1;
distpred = 0:100:9000;
%[~, ii] = sort(dist(routesel));
%routesel = routesel(ii);
[~, ii] = unique(dist(routesel));
routesel = routesel(ii);
%     devpred = interp1(dist(routesel), dev(routesel), distpred);
try
    devpred = spline(dist(routesel), dev(routesel), distpred);
%         plot(dist(routesel), dev(routesel), 'Color', cmap(i, :), 'LineWidth', 2);
    if(~sum(devpred > max(dev) | devpred < min(dev)))
        plot(distpred, devpred, 'Color', cmap(i, :), 'LineWidth', 2);
    end
end
end
xlim([min(dist), max(dist)])
ylim([min(dev), max(dev)])
xlabel('Distance (meters)')
ylabel('Schedule Deviation (seconds)')
% title({'Comparison of Distance Along Trip Computations', 'Color shows distance provided by Puget Sound data feed'})
frame = getframe(gcf);
filename = sprintf('%s/%s_%s_tripInDay_spline_dist_dev.png', imdir, service, route);
imwrite(frame.cdata, filename);

distbins = min(dist):100:max(dist);
distbins(end) = distbins(end) + 0.01;
devbins = min(dev):60:max(dev);
devbins(end) = devbins(end) + 0.01;
N_distbins = length(distbins);
N_devbins = length(devbins);
bins = zeros(N_distbins - 1, N_devbins - 1);
for i_distbin = 1:N_distbins - 1
    for i_devbin = 1:N_devbins - 1
        bins(i_distbin, i_devbin) = sum(...
            dist >= distbins(i_distbin) &...
            dist < distbins(i_distbin + 1) &...
            dev >= devbins(i_devbin) &...
            dev < devbins(i_devbin + 1));
    end
end
distbins(end) = [];
devbins(end) = [];

figure(7); clf; %pcolor(distbins,devbins,bins');
binslogged = log(bins);
adjust = 255/max(binslogged(:))/2;
image(distbins, devbins, binslogged' * adjust);
set(gca, 'YDir', 'normal');
xlabel('Distance (meters)')
ylabel('Schedule Deviation (seconds)')
frame = getframe(gcf);
filename = sprintf('%s/%s_%s_heatmap_dist_dev.png', imdir, service, route);
% hC = colorbar;
% L = [1 2 5 10 20 50 100 200 500 1000 2000 5000 10000];
% l = log(L) * adjust; % Tick mark positions
% set(hC,'Ytick',l,'YTicklabel',L);
imwrite(frame.cdata, filename);



end % function