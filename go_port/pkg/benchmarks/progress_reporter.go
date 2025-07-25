package benchmarks

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// ProgressReporter provides verbose progress reporting for benchmark execution
type ProgressReporter struct {
	mutex          sync.RWMutex
	startTime      time.Time
	totalTasks     int64
	completedTasks int64
	currentTask    string
	lastUpdate     time.Time

	// For ETA calculation
	taskTimes   []time.Duration
	avgTaskTime time.Duration

	// Configuration
	updateInterval      time.Duration
	showETA             bool
	showProgressBar     bool
	showRealTimeResults bool
	progressBarWidth    int

	// Real-time results tracking
	recentResults    []*ProgressResult
	maxRecentResults int

	// State tracking
	isPaused        bool
	pauseStart      time.Time
	totalPausedTime time.Duration
}

// ProgressResult represents a completed benchmark result for real-time display
type ProgressResult struct {
	Timestamp            time.Time `json:"timestamp"`
	AlgorithmCombination string    `json:"algorithm_combination"`
	InputType            string    `json:"input_type"`
	InputSize            int64     `json:"input_size"`
	CompressionRatio     float64   `json:"compression_ratio"`
	ProcessingTimeMs     float64   `json:"processing_time_ms"`
	KEIScore             float64   `json:"kei_score"`
	Status               string    `json:"status"` // "success", "failed", "timeout"
}

// NewProgressReporter creates a new progress reporter
func NewProgressReporter(totalTasks int64) *ProgressReporter {
	return &ProgressReporter{
		startTime:           time.Now(),
		totalTasks:          totalTasks,
		completedTasks:      0,
		updateInterval:      100 * time.Millisecond, // Update every 100ms
		showETA:             true,
		showProgressBar:     true,
		showRealTimeResults: true,
		progressBarWidth:    50,
		maxRecentResults:    10,
		recentResults:       make([]*ProgressResult, 0),
		taskTimes:           make([]time.Duration, 0, 100), // Keep last 100 times for ETA
	}
}

// SetConfiguration configures the progress reporter
func (pr *ProgressReporter) SetConfiguration(updateInterval time.Duration, showETA, showProgressBar, showRealTimeResults bool) {
	pr.mutex.Lock()
	defer pr.mutex.Unlock()

	pr.updateInterval = updateInterval
	pr.showETA = showETA
	pr.showProgressBar = showProgressBar
	pr.showRealTimeResults = showRealTimeResults
}

// StartTask begins tracking a new task
func (pr *ProgressReporter) StartTask(taskName string) {
	pr.mutex.Lock()
	defer pr.mutex.Unlock()

	pr.currentTask = taskName
	pr.lastUpdate = time.Now()

	// Print current task if enough time has passed or if first task
	if pr.completedTasks == 0 || time.Since(pr.lastUpdate) >= pr.updateInterval {
		pr.printProgress()
	}
}

// CompleteTask marks a task as completed and optionally records results
func (pr *ProgressReporter) CompleteTask(result *ProgressResult) {
	pr.mutex.Lock()
	defer pr.mutex.Unlock()

	taskEndTime := time.Now()
	taskDuration := taskEndTime.Sub(pr.lastUpdate)

	// Update task timing statistics
	pr.taskTimes = append(pr.taskTimes, taskDuration)
	if len(pr.taskTimes) > 100 {
		pr.taskTimes = pr.taskTimes[1:] // Keep only last 100
	}

	// Calculate average task time
	totalTime := time.Duration(0)
	for _, t := range pr.taskTimes {
		totalTime += t
	}
	pr.avgTaskTime = totalTime / time.Duration(len(pr.taskTimes))

	pr.completedTasks++

	// Add result to recent results if provided
	if result != nil {
		result.Timestamp = taskEndTime
		pr.recentResults = append(pr.recentResults, result)
		if len(pr.recentResults) > pr.maxRecentResults {
			pr.recentResults = pr.recentResults[1:]
		}
	}

	// Print progress update
	if time.Since(pr.lastUpdate) >= pr.updateInterval || pr.completedTasks%10 == 0 {
		pr.printProgress()
		pr.lastUpdate = taskEndTime
	}
}

// Pause pauses the progress tracking (for user interaction)
func (pr *ProgressReporter) Pause() {
	pr.mutex.Lock()
	defer pr.mutex.Unlock()

	if !pr.isPaused {
		pr.isPaused = true
		pr.pauseStart = time.Now()
		fmt.Println("\n[PAUSED] Benchmark execution paused...")
	}
}

// Resume resumes the progress tracking
func (pr *ProgressReporter) Resume() {
	pr.mutex.Lock()
	defer pr.mutex.Unlock()

	if pr.isPaused {
		pr.totalPausedTime += time.Since(pr.pauseStart)
		pr.isPaused = false
		fmt.Println("[RESUMED] Benchmark execution resumed...")
		pr.printProgress()
	}
}

// printProgress displays the current progress (called with mutex held)
func (pr *ProgressReporter) printProgress() {
	if pr.isPaused {
		return
	}

	// Clear line and move cursor to beginning
	fmt.Print("\r\033[K")

	// Calculate progress percentage
	progressPercent := float64(pr.completedTasks) / float64(pr.totalTasks) * 100.0

	// Build progress display
	var display strings.Builder

	// Progress bar
	if pr.showProgressBar {
		display.WriteString(pr.buildProgressBar(progressPercent))
		display.WriteString(" ")
	}

	// Percentage and counts
	display.WriteString(fmt.Sprintf("%.1f%% (%d/%d)",
		progressPercent, pr.completedTasks, pr.totalTasks))

	// Current task
	if pr.currentTask != "" {
		display.WriteString(fmt.Sprintf(" | %s", pr.currentTask))
	}

	// ETA
	if pr.showETA && pr.avgTaskTime > 0 {
		remainingTasks := pr.totalTasks - pr.completedTasks
		estimatedTimeRemaining := time.Duration(remainingTasks) * pr.avgTaskTime
		display.WriteString(fmt.Sprintf(" | ETA: %s", pr.formatDuration(estimatedTimeRemaining)))
	}

	// Elapsed time
	elapsedTime := time.Since(pr.startTime) - pr.totalPausedTime
	display.WriteString(fmt.Sprintf(" | Elapsed: %s", pr.formatDuration(elapsedTime)))

	fmt.Print(display.String())

	// Print recent results if enabled
	if pr.showRealTimeResults && len(pr.recentResults) > 0 {
		pr.printRecentResults()
	}
}

// buildProgressBar creates a text-based progress bar
func (pr *ProgressReporter) buildProgressBar(percent float64) string {
	filledWidth := int(percent / 100.0 * float64(pr.progressBarWidth))
	if filledWidth > pr.progressBarWidth {
		filledWidth = pr.progressBarWidth
	}

	bar := strings.Builder{}
	bar.WriteString("[")

	for i := 0; i < pr.progressBarWidth; i++ {
		if i < filledWidth {
			bar.WriteString("█")
		} else {
			bar.WriteString("░")
		}
	}

	bar.WriteString("]")
	return bar.String()
}

// printRecentResults displays recent benchmark results
func (pr *ProgressReporter) printRecentResults() {
	if len(pr.recentResults) == 0 {
		return
	}

	fmt.Println() // New line after progress bar

	// Header
	fmt.Println("Recent Results:")
	fmt.Println("Time     | Combination                    | Input Type    | Size   | Ratio | Time(ms) | KEI   | Status")
	fmt.Println("---------|--------------------------------|---------------|--------|-------|----------|-------|--------")

	// Show last few results
	start := 0
	if len(pr.recentResults) > 5 {
		start = len(pr.recentResults) - 5
	}

	for i := start; i < len(pr.recentResults); i++ {
		result := pr.recentResults[i]
		timeStr := result.Timestamp.Format("15:04:05")

		// Truncate long combination names
		combo := result.AlgorithmCombination
		if len(combo) > 30 {
			combo = combo[:27] + "..."
		}

		// Format input size
		sizeStr := pr.formatBytes(result.InputSize)

		// Status indicator
		statusIcon := "✓"
		switch result.Status {
		case "failed":
			statusIcon = "✗"
		case "timeout":
			statusIcon = "⏱"
		}

		fmt.Printf("%s | %-30s | %-13s | %-6s | %5.2f | %8.1f | %5.1f | %s\n",
			timeStr, combo, result.InputType, sizeStr,
			result.CompressionRatio, result.ProcessingTimeMs, result.KEIScore, statusIcon)
	}

	fmt.Print("\n") // Add space before next progress update
}

// formatBytes formats byte count into human-readable format
func (pr *ProgressReporter) formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%dB", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f%cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// formatDuration formats a duration into human-readable format
func (pr *ProgressReporter) formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	} else if d < time.Hour {
		minutes := int(d.Minutes())
		seconds := int(d.Seconds()) % 60
		return fmt.Sprintf("%dm%ds", minutes, seconds)
	} else {
		hours := int(d.Hours())
		minutes := int(d.Minutes()) % 60
		return fmt.Sprintf("%dh%dm", hours, minutes)
	}
}

// PrintFinalSummary prints a final summary when benchmarking is complete
func (pr *ProgressReporter) PrintFinalSummary() {
	pr.mutex.RLock()
	defer pr.mutex.RUnlock()

	fmt.Printf("\r\033[K") // Clear current line
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("                    BENCHMARK COMPLETE")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	totalTime := time.Since(pr.startTime) - pr.totalPausedTime

	fmt.Printf("Total tasks completed: %d/%d\n", pr.completedTasks, pr.totalTasks)
	fmt.Printf("Total execution time: %s\n", pr.formatDuration(totalTime))

	if pr.totalPausedTime > 0 {
		fmt.Printf("Total paused time: %s\n", pr.formatDuration(pr.totalPausedTime))
	}

	if len(pr.taskTimes) > 0 {
		fmt.Printf("Average task time: %s\n", pr.formatDuration(pr.avgTaskTime))

		// Calculate tasks per second
		tasksPerSecond := float64(pr.completedTasks) / totalTime.Seconds()
		fmt.Printf("Average throughput: %.2f tasks/second\n", tasksPerSecond)
	}

	if len(pr.recentResults) > 0 {
		// Summary statistics
		var totalKEI, totalRatio, totalTime float64
		successCount := 0

		for _, result := range pr.recentResults {
			if result.Status == "success" {
				totalKEI += result.KEIScore
				totalRatio += result.CompressionRatio
				totalTime += result.ProcessingTimeMs
				successCount++
			}
		}

		if successCount > 0 {
			fmt.Printf("\nRecent Results Summary (last %d results):\n", len(pr.recentResults))
			fmt.Printf("  Average KEI Score: %.2f\n", totalKEI/float64(successCount))
			fmt.Printf("  Average Compression Ratio: %.2f\n", totalRatio/float64(successCount))
			fmt.Printf("  Average Processing Time: %.1f ms\n", totalTime/float64(successCount))
			fmt.Printf("  Success Rate: %.1f%% (%d/%d)\n",
				float64(successCount)/float64(len(pr.recentResults))*100,
				successCount, len(pr.recentResults))
		}
	}

	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()
}

// GetProgress returns current progress information
func (pr *ProgressReporter) GetProgress() map[string]interface{} {
	pr.mutex.RLock()
	defer pr.mutex.RUnlock()

	progressPercent := float64(pr.completedTasks) / float64(pr.totalTasks) * 100.0
	elapsedTime := time.Since(pr.startTime) - pr.totalPausedTime

	var estimatedTimeRemaining time.Duration
	if pr.avgTaskTime > 0 {
		remainingTasks := pr.totalTasks - pr.completedTasks
		estimatedTimeRemaining = time.Duration(remainingTasks) * pr.avgTaskTime
	}

	return map[string]interface{}{
		"total_tasks":          pr.totalTasks,
		"completed_tasks":      pr.completedTasks,
		"progress_percent":     progressPercent,
		"current_task":         pr.currentTask,
		"elapsed_time":         elapsedTime.String(),
		"estimated_remaining":  estimatedTimeRemaining.String(),
		"is_paused":            pr.isPaused,
		"tasks_per_second":     float64(pr.completedTasks) / elapsedTime.Seconds(),
		"recent_results_count": len(pr.recentResults),
	}
}

// SetTotalTasks updates the total number of tasks (useful if count changes during execution)
func (pr *ProgressReporter) SetTotalTasks(total int64) {
	pr.mutex.Lock()
	defer pr.mutex.Unlock()

	pr.totalTasks = total
}

// PrintTaskStarted prints a verbose message when starting a significant task
func (pr *ProgressReporter) PrintTaskStarted(taskDescription string) {
	fmt.Printf("\n🚀 Starting: %s\n", taskDescription)
}

// PrintTaskCompleted prints a verbose message when completing a significant task
func (pr *ProgressReporter) PrintTaskCompleted(taskDescription string, duration time.Duration) {
	fmt.Printf("✅ Completed: %s (took %s)\n", taskDescription, pr.formatDuration(duration))
}

// PrintError prints an error message with proper formatting
func (pr *ProgressReporter) PrintError(operation string, err error) {
	fmt.Printf("❌ Error in %s: %v\n", operation, err)
}

// PrintWarning prints a warning message
func (pr *ProgressReporter) PrintWarning(message string) {
	fmt.Printf("⚠️  Warning: %s\n", message)
}

// PrintInfo prints an informational message
func (pr *ProgressReporter) PrintInfo(message string) {
	fmt.Printf("ℹ️  %s\n", message)
}
