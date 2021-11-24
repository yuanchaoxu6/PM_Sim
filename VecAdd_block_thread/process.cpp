#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
using namespace std;

int main()
{
    FILE *out=fopen("res", "w+");
    int i;
    vector <double> v[20];
    for (i = 1; i <= 160; )
    {
        char str[100];
        sprintf(str, "%d", i);
        FILE *fp=fopen(str,"r");
        
        char buf[100];
        int index[200];
        memset(index,0,sizeof(index));
        for (int j = 0; j < 20; ++j) v[j].clear();
        while (fgets(buf,80,fp)!=NULL)
        {
            double value;
            int tid;
            if (buf[0] != 'T') continue;
            //printf("%s\n", buf);
            sscanf(buf,"TID: %d BW: %lf", &tid, &value);
            index[tid]++;
            v[index[tid]-1].push_back(value);
        }
        for (int j = 0; j < 20; ++j)
            sort(v[j].begin(), v[j].end());
        for (int j = 0; j < 20; ++j)
        {
            for (int k = 0; k < v[j].size(); ++k)
                fprintf(out, "%lf ", v[j][k]);
            fprintf(out, "\n");
        }
        if (i<=16) i=i*2;
        else i=i+16;
    }
    fclose(out);
    return 0;
}