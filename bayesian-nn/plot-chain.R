my_data1 <- read.table("chain_p11_scabpit.txt", header = FALSE, sep = "", dec = ".")
a1 <- my_data1[13:100000, ]
b1 <- my_data1[100013:200000, ]
c1 <- my_data1[200013:300000, ]
d1 <- my_data1[300013:400000, ]
e1 <- my_data1[400013:500000, ]
f1 <- my_data1[500013:600000, ]
g1 <- my_data1[600013:700000, ]
h1 <- my_data1[700013:800000, ]
i1 <- my_data1[800013:900000, ]

plot(a1$V1, a1$V2, type = "l", col = "red")
plot(b1$V1, b1$V2, type = "l", col = "red")
plot(c1$V1, c1$V2, type = "l", col = "red")
plot(d1$V1, d1$V2, type = "l", col = "red")
plot(e1$V1, e1$V2, type = "l", col = "red")
plot(f1$V1, f1$V2, type = "l", col = "red")
plot(g1$V1, g1$V2, type = "l", col = "red")
plot(h1$V1, h1$V2, type = "l", col = "red")
plot(i1$V1, i1$V2, type = "l", col = "red")

my_data2 <- read.table("p11_scabpit_large.txt", header = FALSE, sep = "", dec = ".")
a2 <- my_data2[5:1000,]
b2 <- my_data2[1005:2000, ]
c2 <- my_data2[2005:3000, ]
d2 <- my_data2[3005:4000, ]
e2 <- my_data2[4005:5000, ]
f2 <- my_data2[5005:6000, ]
g2 <- my_data2[6005:7000, ]
h2 <- my_data2[7005:8000, ]
i2 <- my_data2[8005:9000, ]
j2 <- my_data2[9005:10000, ]
k2 <- my_data2[10005:11000, ]

#1->h0
plot(a2$V1, a2$V2, type = "l", col = "red")
#bh0
plot(b2$V1, b2$V2, type = "l", col = "red")
#h0->h1
plot(c2$V1, c2$V2, type = "l", col = "red")
#bh1
plot(d2$V1, d2$V2, type = "l", col = "red")
#h1->h2
plot(e2$V1, e2$V2, type = "l", col = "red")
#bh2
plot(f2$V1, f2$V2, type = "l", col = "red")
#h2->h3
plot(g2$V1, g2$V2, type = "l", col = "red")
#bh3
plot(h2$V1, h2$V2, type = "l", col = "red")
#h3->h4
plot(i2$V1, i2$V2, type = "l", col = "red")
#bh4
plot(j2$V1, j2$V2, type = "l", col = "red")
#h4->o
plot(k2$V1, k2$V2, type = "l", col = "red")

my_data3 <- read.table("plot_3.txt", header = FALSE, sep = "", dec = ".")
a3 <- my_data3[5:10000,]
b3 <- my_data3[10005:20000, ]
c3 <- my_data3[20005:30000, ]
d3 <- my_data3[30005:40000, ]
e3 <- my_data3[40005:50000, ]
f3 <- my_data3[50005:60000, ]
g3 <- my_data3[60005:70000, ]
h3 <- my_data3[70005:80000, ]
i3 <- my_data3[80005:90000, ]

#input -> h0
plot(a1$V1, a1$V2, type = "l", col = "red")
lines(a2$V1, a2$V2, type = "l", col = "blue")

#bh0
plot(b1$V1, b1$V2, type = "l", col = "red")
lines(b2$V1, b2$V2, type = "l", col = "blue")

#h0 -> h1
plot(c1$V1, c1$V2, type = "l", col = "red")
lines(c2$V1, c2$V2, type = "l", col = "blue")

#bh1
plot(d1$V1, d1$V2, type = "l", col = "red")
lines(d2$V1, d2$V2, type = "l", col = "blue")

#h1 -> h2
plot(e1$V1, e1$V2, type = "l", col = "red")
lines(e2$V1, e2$V2, type = "l", col = "blue")
#lines(e1$V1, e1$V2, type = "l", col = "green")

#bh2
plot(f1$V1, f1$V2, type = "l", col = "red")
lines(f2$V1, f2$V2, type = "l", col = "blue")
#lines(f1$V1, f1$V2, type = "l", col = "green")

#h2 -> h3
plot(g1$V1, g1$V2, type = "l", col = "red")
lines(g2$V1, g2$V2, type = "l", col = "blue")

#bh3
plot(h1$V1, h1$V2, type = "l", col = "red")
lines(h2$V1, h2$V2, type = "l", col = "blue")
#lines(h1$V1, h1$V2, type = "l", col = "green")

#h3 -> output
plot(i1$V1, i1$V2, type = "l", col = "red")
lines(i2$V1, i2$V2, type = "l", col = "blue")
#lines(i1$V1, i1$V2, type = "l", col = "green")

