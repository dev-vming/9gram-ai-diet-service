import { PipeTransform, Injectable, ArgumentMetadata, BadRequestException } from '@nestjs/common';
// import { Food } from '../record.entity';

@Injectable()
export class ValidateFoodPipe implements PipeTransform {
  transform(value: any, metadata: ArgumentMetadata) {
    if (metadata.type === 'body') {
      if (!value.foods.every(food => this.isValidFood(food))) {
        throw new BadRequestException('유효하지 않은 음식 데이터가 포함되어 있습니다.');
      }
    }
    return value;
  }

  private isValidFood(food: any): boolean {
    return food.hasOwnProperty('foodName') && typeof food.foodName === 'string' &&
           food.hasOwnProperty('counts') && typeof food.counts === 'number' &&
           food.hasOwnProperty('XYCoordinate') && typeof food.XYCoordinate === 'object';
  }
}