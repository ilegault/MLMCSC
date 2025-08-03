# Nginx Dockerfile for MLMCSC
FROM nginx:1.25-alpine

# Install additional packages
RUN apk add --no-cache \
    curl \
    openssl

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf
COPY default.conf /etc/nginx/conf.d/default.conf

# Create SSL directory
RUN mkdir -p /etc/nginx/ssl

# Copy static files
COPY --from=mlmcsc-api:latest /app/src/web/static /app/static

# Create nginx user and set permissions
RUN addgroup -g 101 -S nginx && \
    adduser -S -D -H -u 101 -h /var/cache/nginx -s /sbin/nologin -G nginx -g nginx nginx && \
    chown -R nginx:nginx /var/cache/nginx /var/log/nginx /etc/nginx

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/nginx_status || exit 1

# Expose ports
EXPOSE 80 443 8080

# Start nginx
CMD ["nginx", "-g", "daemon off;"]