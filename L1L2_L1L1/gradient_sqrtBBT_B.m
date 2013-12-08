function G=gradient_sqrtBBT_B(B,T)

G_bbt=gradientsqrtBBt(B,T);


G=(B*B')^(-1/2)*T+G_bbt*B;
end