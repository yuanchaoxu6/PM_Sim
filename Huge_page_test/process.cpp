#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
using namespace std;

int main()
{
    FILE *out=fopen("res", "w+");
    int i;
    for (i = 1; i <= 10; i++)
    {
        char str[100];
        sprintf(str, "res1", i);
        FILE *fp=fopen(str,"r");
        vector <double> v;
        char buf[100];
        while (fgets(buf,80,fp)!=NULL)
        {
            double sizes, stride, time;
            sscanf(buf, "%lf%lf%lf", &sizes, &stride, &time);
            v.push_back(time);
        }
        int i, j, k;
        j = 0; k = 7;
        for (i = 0; i < v.size(); i++)
        {
            printf("%lf ", v[i]*1000.0/1024.0/1024.0/128.0);
            j++;
            if (j == k) {printf("\n"); j = -1; k++;}
        }
    }
    fclose(out);
    return 0;
}