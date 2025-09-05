# VectorQA Sage - Makefile Deployment System
# Auth: Rudy Martin - Next Shift Consulting LLC
# Date: 2025-08-23

.PHONY: help init full clean status logs stop test

# Default target
help:
	@echo "VectorQA Sage - Deployment Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make init          - Deploy initial test app (connection testing)"
	@echo "  make full          - Deploy full application"
	@echo "  make test          - Run tests on current deployment"
	@echo "  make status        - Check deployment status"
	@echo "  make logs          - Show application logs"
	@echo "  make stop          - Stop all applications"
	@echo "  make clean         - Clean up all deployments and containers"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Deployment workflow:"
	@echo "  1. make init       # Test connections and environment"
	@echo "  2. make test       # Verify everything works"
	@echo "  3. make full       # Deploy full application"
	@echo "  4. make status     # Monitor deployment"
	@echo ""

# Deploy initial test app (connection testing)
init:
	@echo "🚀 Deploying initial test app..."
	@./progressive_deploy.sh minimal
	@echo ""
	@echo "✅ Initial app deployed!"
	@echo "📱 Access test app at: http://localhost:8501"
	@echo "🔍 Verify all tests pass before running 'make full'"

# Deploy full application
full:
	@echo "🚀 Deploying full application..."
	@./progressive_deploy.sh full
	@echo ""
	@echo "✅ Full application deployed!"
	@echo "📱 Access application at: http://localhost:8501"
	@echo "📊 Run 'make status' to check health"

# Run tests on current deployment
test:
	@echo "🧪 Running deployment tests..."
	@if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then \
		echo "✅ Application is running"; \
		if curl -s http://localhost:8501 | grep -q "Connection Test"; then \
			echo "📋 Running minimal app tests..."; \
			echo "   - Check browser at http://localhost:8501"; \
			echo "   - Verify all tests show ✅ (green)"; \
			echo "   - Fix any ❌ errors before 'make full'"; \
		else \
			echo "📋 Running full app tests..."; \
			echo "   - Check browser at http://localhost:8501"; \
			echo "   - Verify all features work correctly"; \
			echo "   - Check logs with 'make logs' if issues"; \
		fi; \
	else \
		echo "❌ No application is running"; \
		echo "💡 Run 'make init' to start deployment"; \
		exit 1; \
	fi

# Check deployment status
status:
	@echo "📊 Checking deployment status..."
	@./progressive_deploy.sh status

# Show application logs
logs:
	@echo "📋 Showing application logs..."
	@if command -v docker-compose >/dev/null 2>&1; then \
		docker-compose logs -f --tail=50; \
	else \
		if [ -f "../app/app.log" ]; then \
			tail -f ../app/app.log; \
		elif [ -f "../app/minimal.log" ]; then \
			tail -f ../app/minimal.log; \
		else \
			echo "❌ No log files found"; \
		fi; \
	fi

# Stop all applications
stop:
	@echo "🛑 Stopping all applications..."
	@./progressive_deploy.sh stop
	@echo "✅ All applications stopped"

# Clean up all deployments and containers
clean:
	@echo "🧹 Cleaning up deployments..."
	@echo "Stopping applications..."
	@./progressive_deploy.sh stop 2>/dev/null || true
	@echo "Removing containers..."
	@docker-compose down --volumes --remove-orphans 2>/dev/null || true
	@docker-compose -f docker-compose.minimal.yml down --volumes --remove-orphans 2>/dev/null || true
	@echo "Removing log files..."
	@rm -f ../app/app.log ../app/minimal.log 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Quick deployment (init + test + full)
deploy: init
	@echo ""
	@echo "⏳ Waiting for tests to complete..."
	@sleep 5
	@echo "🧪 Running tests..."
	@make test
	@echo ""
	@echo "🚀 Deploying full application..."
	@make full
	@echo ""
	@echo "🎉 Deployment complete!"
	@echo "📱 Access application at: http://localhost:8501"

# Development mode (with auto-reload)
dev:
	@echo "🔧 Starting development mode..."
	@echo "This will start the app with auto-reload for development"
	@cd ../app && streamlit run src/main.py --server.port=8501 --server.address=0.0.0.0 --server.runOnSave=true

# Production mode (optimized)
prod:
	@echo "🏭 Starting production mode..."
	@echo "This will start the app with production optimizations"
	@make full
	@echo "✅ Production deployment ready"

# Health check
health:
	@echo "🏥 Running health check..."
	@if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then \
		echo "✅ Application is healthy"; \
		echo "📊 Health endpoint: http://localhost:8501/_stcore/health"; \
	else \
		echo "❌ Application health check failed"; \
		echo "💡 Run 'make status' for more details"; \
		exit 1; \
	fi

# Update deployment (stop + full)
update:
	@echo "🔄 Updating deployment..."
	@make stop
	@make full
	@echo "✅ Update complete"

# Backup current deployment
backup:
	@echo "💾 Creating backup..."
	@if [ -d "../app" ]; then \
		tar -czf "backup_$(date +%Y%m%d_%H%M%S).tar.gz" -C .. app environ_settings; \
		echo "✅ Backup created: backup_$(date +%Y%m%d_%H%M%S).tar.gz"; \
	else \
		echo "❌ No app directory found to backup"; \
	fi

# Restore from backup
restore:
	@echo "📦 Restoring from backup..."
	@if [ -f "backup_*.tar.gz" ]; then \
		BACKUP_FILE=$$(ls -t backup_*.tar.gz | head -1); \
		echo "Restoring from: $$BACKUP_FILE"; \
		tar -xzf "$$BACKUP_FILE" -C ..; \
		echo "✅ Restore complete"; \
	else \
		echo "❌ No backup files found"; \
	fi
