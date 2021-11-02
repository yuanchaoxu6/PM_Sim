#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
using namespace std;

int main()
{
    FILE *out=fopen("res", "w+");
    int i;
    for (i = 1; i <= 160; )
    {
        char str[100];
        sprintf(str, "%d", i);
        FILE *fp=fopen(str,"r");
        vector <double> v;
        char buf[100];
        while (fgets(buf,80,fp)!=NULL)
        {
            double value;
            sscanf(buf,"BW: %lf", &value);
            v.push_back(value);
        }
        double avg = 0.0;
        for (int j = v.size()-1; j > v.size()-11; --j)
            avg+=v[j];
        fprintf(out,"%lf\n", avg/10.0);
        if (i<=16) i=i*2;
        else i=i+16;
    }
    fclose(out);
    return 0;
}