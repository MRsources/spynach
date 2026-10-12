% Writes the top functions of a MATLAB profiler run as JSON:
%
%     cmp_profile(file,profile('info'),top_n)
%
% Two lists, by total time (including callees) and by self time (total
% minus the time of the children), each with function, file, calls,
% total_s and self_s. Only functions under the Spinach directory and
% the comparison case are kept in the self-time list, which makes the
% kernel hot spots visible; MATLAB built-ins are kept in the total list.
function cmp_profile(file,info,top_n)
ft=info.FunctionTable;
if isempty(ft)
    cmp_json(file,struct('by_total',{{}},'by_self',{{}})); return;
end
total=[ft.TotalTime];
self=zeros(size(total));
for n=1:numel(ft)
    children=ft(n).Children;
    if isempty(children), self(n)=total(n); else, self(n)=total(n)-sum([children.TotalTime]); end
end
records=struct('function',{ft.FunctionName},'file',{ft.FileName},...
               'calls',num2cell([ft.NumCalls]),'total_s',num2cell(total),'self_s',num2cell(self));
[~,order]=sort(total,'descend');
by_total=records(order(1:min(top_n,numel(order))));
[~,order]=sort(self,'descend');
by_self=records(order(1:min(top_n,numel(order))));
cmp_json(file,struct('by_total',{by_total},'by_self',{by_self}));   % cells: struct arrays would broadcast
end
