from data_loader import load_data_simple, split_data

def main():
    X, y = load_data_simple()
    print(X.shape)
    print(y.shape)
    iterations = 5
    stride = 1/iterations

    for iteration in range(iterations):
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, iteration, stride, split=[0.70, 0.15, 0.15])

        print(X_train.shape)
        print(y_train.shape)
        print(X_val.shape)
        print(y_val.shape)
        print(X_test.shape)
        print(y_test.shape)



if __name__ == '__main__':
    main()