import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Error caught in ErrorBoundary:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="bg-red-600 text-white p-4 rounded-lg shadow-lg m-4">
          <h2 className="text-lg font-bold">خطایی رخ داده است</h2>
          <p>{this.state.error?.message || "لطفاً صفحه را رفرش کنید یا با پشتیبانی تماس بگیرید."}</p>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;