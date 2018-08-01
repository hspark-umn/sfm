function VisualizeSIFT

im_str = sprintf('image%07d.bmp', 1);
im_key = sprintf('image%07d.key', 1);
key = LoadKey(im_key);
im = imread(im_str);

figure(1)
clf;
imshow(im);
hold on
plot(key(:,2), key(:,1), 'rx');


function key = LoadKey(filename)
fid = fopen(filename);
n = fscanf(fid, '%d', 1);
fscanf(fid, '%d', 1);
X = fscanf(fid, '%f', [132 n])';
key = X(:, 1:2);

fclose(fid);